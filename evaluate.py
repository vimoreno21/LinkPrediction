import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

def compute_edge_features_optimized(G, G_undirected, degrees, neighbors, preds, succs, node_u, node_v, node_features):
    u_feats = node_features.get(node_u, {})
    v_feats = node_features.get(node_v, {})

    # Common neighbors & Jaccard
    u_nbrs = neighbors.get(node_u, set())
    v_nbrs = neighbors.get(node_v, set())

    common = u_nbrs & v_nbrs
    union = u_nbrs | v_nbrs
    num_common_neighbors = len(common)
    jaccard = len(common) / len(union) if union else 0.0

    # Preferential attachment
    pref_attachment = degrees.get(node_u, 0) * degrees.get(node_v, 0)

    # Adamic-Adar
    adamic_adar = sum(1.0 / np.log(degrees[n]) for n in common if degrees.get(n, 0) > 1)

    # Directed features
    u_preds = preds.get(node_u, set())
    v_preds = preds.get(node_v, set())
    u_succs = succs.get(node_u, set())
    v_succs = succs.get(node_v, set())

    reciprocity = 1.0 if node_u in v_succs else 0.0
    follower_overlap = len(u_preds & v_preds) / max(1, min(len(u_preds), len(v_preds)))
    following_overlap = len(u_succs & v_succs) / max(1, min(len(u_succs), len(v_succs)))

    return {
        'u_degree': u_feats.get('degree_centrality', 0.0),
        'u_in_degree': u_feats.get('in_degree_centrality', 0.0),
        'u_out_degree': u_feats.get('out_degree_centrality', 0.0),
        'u_pagerank': u_feats.get('pagerank', 0.0),
        'u_clustering': u_feats.get('clustering', 0.0),
        'v_degree': v_feats.get('degree_centrality', 0.0),
        'v_in_degree': v_feats.get('in_degree_centrality', 0.0),
        'v_out_degree': v_feats.get('out_degree_centrality', 0.0),
        'v_pagerank': v_feats.get('pagerank', 0.0),
        'v_clustering': v_feats.get('clustering', 0.0),
        'common_neighbors': num_common_neighbors,
        'preferential_attachment': pref_attachment,
        'jaccard_coefficient': jaccard,
        'adamic_adar': adamic_adar,
        'reciprocity': reciprocity,
        'follower_overlap': follower_overlap,
        'following_overlap': following_overlap
    }


def compute_edge_features(G, node_u, node_v, node_features):
    """
    Compute features for a potential edge between nodes u and v

    This function creates a comprehensive feature vector for a potential connection between
    two nodes, combining individual node characteristics and relationship metrics.

    Feature Categories:
    1. Node-specific features for both endpoints (from node_features)
    2. Topological features measuring similarity or connection strength
    3. Directed relationship features specific to follower networks

    Args:
        G: NetworkX DiGraph - The network graph
        node_u: Source node ID
        node_v: Target node ID
        node_features: Dictionary of node features from extract_topological_features()

    Returns:
        Dictionary of edge features
    """
    # --- 1. Node-specific Features ---

    # Get pre-calculated features for both nodes
    u_feats = node_features.get(node_u, {})
    v_feats = node_features.get(node_v, {})

    # --- 2. Topological Connection Features ---

    # For topological measures, we use undirected version of the graph
    # This captures general connectivity patterns regardless of direction
    G_undirected = G.to_undirected()

    # Common neighbors: number of shared connections between u and v
    # Higher values suggest nodes in the same community or with similar interests
    try:
        common_neighbors = list(nx.common_neighbors(G_undirected, node_u, node_v))
        num_common_neighbors = len(common_neighbors)
    except:
        num_common_neighbors = 0

    # Preferential attachment: product of node degrees
    # Based on the idea that highly-connected nodes are more likely to form new connections
    pref_attachment = G_undirected.degree(node_u) * G_undirected.degree(node_v)

    # Jaccard coefficient: proportion of common neighbors relative to all neighbors
    # Normalizes common neighbors by total neighborhood size
    try:
        u_neighbors = set(G_undirected.neighbors(node_u))
        v_neighbors = set(G_undirected.neighbors(node_v))
        if len(u_neighbors | v_neighbors) > 0:
            jaccard = len(u_neighbors & v_neighbors) / len(u_neighbors | v_neighbors)
        else:
            jaccard = 0.0
    except:
        jaccard = 0.0

    # --- 3. Directed Relationship Features ---

    # Features specific to directed networks like Twitter
    try:
        # Get followers and followees for both nodes
        u_successors = set(G.successors(node_u))  # Users that u follows
        u_predecessors = set(G.predecessors(node_u))  # Users that follow u
        v_successors = set(G.successors(node_v))  # Users that v follows
        v_predecessors = set(G.predecessors(node_v))  # Users that follow v

        # Reciprocity: whether v already follows u (indicating potential for reciprocation)
        # Binary feature: 1.0 if v follows u, 0.0 otherwise
        reciprocity = 1.0 if node_u in v_successors else 0.0

        # Follower overlap: proportion of shared followers relative to the smaller follower set
        # Higher values indicate users with similar audiences
        follower_overlap = len(u_predecessors & v_predecessors) / max(1, min(len(u_predecessors), len(v_predecessors)))

        # Following overlap: proportion of shared followees relative to the smaller following set
        # Higher values indicate users with similar interests
        following_overlap = len(u_successors & v_successors) / max(1, min(len(u_successors), len(v_successors)))
    except:
        # Default values if calculation fails
        reciprocity = 0.0
        follower_overlap = 0.0
        following_overlap = 0.0

    # Adamic-Adar index: weighted common neighbors (higher weight for rare connections)
    # Gives more importance to common neighbors that have fewer connections themselves
    adamic_adar = 0.0
    try:
        for common_neighbor in common_neighbors:
            neighbor_degree = G_undirected.degree(common_neighbor)
            if neighbor_degree > 1:  # Avoid log(1) = 0 division
                adamic_adar += 1.0 / np.log(neighbor_degree)
    except:
        pass

    # --- Combine All Features Into a Single Vector ---

    edge_features = {
        # Node features for source node (u)
        'u_degree': u_feats.get('degree_centrality', 0.0),
        'u_in_degree': u_feats.get('in_degree_centrality', 0.0),
        'u_out_degree': u_feats.get('out_degree_centrality', 0.0),
        'u_pagerank': u_feats.get('pagerank', 0.0),
        'u_clustering': u_feats.get('clustering', 0.0),

        # Node features for target node (v)
        'v_degree': v_feats.get('degree_centrality', 0.0),
        'v_in_degree': v_feats.get('in_degree_centrality', 0.0),
        'v_out_degree': v_feats.get('out_degree_centrality', 0.0),
        'v_pagerank': v_feats.get('pagerank', 0.0),
        'v_clustering': v_feats.get('clustering', 0.0),

        # Edge-specific topological features
        'common_neighbors': num_common_neighbors,
        'preferential_attachment': pref_attachment,
        'jaccard_coefficient': jaccard,
        'adamic_adar': adamic_adar,

        # Directed relationship features
        'reciprocity': reciprocity,
        'follower_overlap': follower_overlap,
        'following_overlap': following_overlap
    }

    return edge_features



def evaluate_model(y_true, y_scores):
    """
    Evaluate model performance using classification metrics

    This function calculates and visualizes key performance metrics for the link prediction model:
    - ROC AUC: Area under the Receiver Operating Characteristic curve
    - Average Precision: Area under the Precision-Recall curve

    Args:
        y_true: numpy.ndarray - Ground truth labels (1 for positive edges, 0 for negative edges)
        y_scores: numpy.ndarray - Predicted probability scores from the model

    Returns:
        Tuple containing (auc_score, ap_score)
    """
    # ---------- CLASSIFICATION METRICS ----------

    # ROC AUC score - measures the model's ability to discriminate between classes
    # Higher values indicate better separation between positive and negative examples
    # 0.5 = random guessing, 1.0 = perfect separation
    auc_score = roc_auc_score(y_true, y_scores)

    # Precision-Recall curve data points
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Average Precision - summarizes the precision-recall curve as the weighted mean of precisions at each threshold
    # Higher values indicate better performance across different threshold settings
    ap_score = average_precision_score(y_true, y_scores)

    # Print the metrics
    print(f"Performance metrics:")
    print(f"  - ROC AUC Score: {auc_score:.4f}")
    print(f"  - Average Precision Score: {ap_score:.4f}")

    # ---------- VISUALIZATION: PRECISION-RECALL CURVE ----------

    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', label=f'Average Precision = {ap_score:.4f}')

    # Add a baseline (random classifier)
    # For a balanced dataset, this would be the positive class proportion
    positive_ratio = np.mean(y_true)
    plt.axhline(y=positive_ratio, color='r', linestyle='--', alpha=0.5,
               label=f'Random Classifier (AP = {positive_ratio:.4f})')

    # Add chart decorations
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Link Prediction')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    # Save the figure
    plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- VISUALIZATION: ROC CURVE ----------

    from sklearn.metrics import roc_curve

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')

    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier (AUC = 0.5)')

    # Add chart decorations
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Link Prediction')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    # Save the figure
    plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- VISUALIZATION: THRESHOLD ANALYSIS ----------

    # Create a dataframe with different threshold values and their corresponding metrics
    threshold_metrics = []

    # Exclude the last threshold which is often 0 and leads to division by zero
    for i in range(len(thresholds)):
        # Predictions at current threshold
        y_pred = (y_scores >= thresholds[i]).astype(int)

        # Calculate metrics
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        false_positives = np.sum((y_pred == 1) & (y_true == 0))
        true_negatives = np.sum((y_pred == 0) & (y_true == 0))
        false_negatives = np.sum((y_pred == 0) & (y_true == 1))

        # Handle division by zero
        if true_positives + false_positives == 0:
            prec = 0
        else:
            prec = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            rec = 0
        else:
            rec = true_positives / (true_positives + false_negatives)

        # Calculate F1 score
        if prec + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)

        # Calculate accuracy
        acc = (true_positives + true_negatives) / len(y_true)

        threshold_metrics.append({
            'Threshold': thresholds[i],
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'Accuracy': acc,
            'True Positives': true_positives,
            'False Positives': false_positives
        })

    # Convert to dataframe
    threshold_df = pd.DataFrame(threshold_metrics)

    # Plot metrics vs threshold
    plt.figure(figsize=(12, 8))

    plt.plot(threshold_df['Threshold'], threshold_df['Precision'], 'b-', label='Precision')
    plt.plot(threshold_df['Threshold'], threshold_df['Recall'], 'g-', label='Recall')
    plt.plot(threshold_df['Threshold'], threshold_df['F1 Score'], 'r-', label='F1 Score')
    plt.plot(threshold_df['Threshold'], threshold_df['Accuracy'], 'y-', label='Accuracy')

    # Add chart decorations
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metric Performance vs. Classification Threshold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    # Save the figure
    plt.savefig("threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Find threshold with highest F1
    best = max(threshold_metrics, key=lambda x: x['F1 Score'])

    return auc_score, ap_score, best['Precision'], best['Recall'], best['F1 Score'], best['Accuracy']

