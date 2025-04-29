import os
import random
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve, 
                            roc_curve, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
from node2vec import Node2Vec

# ================================
# Utility Functions
# ================================

def load_ego_network(ego_id, data_dir):
    """
    Load a Twitter ego network from the specified directory.
    
    Args:
        ego_id: ID of the ego network to load
        data_dir: Directory containing the Twitter ego networks
    
    Returns:
        NetworkX graph of the ego network
    """
    ego_file = os.path.join(data_dir, "twitter", f"{ego_id}.edges")
    
    try:
        # Using undirected graph to avoid common_neighbors issues with directed graphs
        G = nx.read_edgelist(ego_file, create_using=nx.Graph(), nodetype=int)
        print(f"Loaded ego network {ego_id} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"Error loading ego network {ego_id}: {str(e)}")
        return None

def create_union_graph(data_dir, num_egos=50, edges_per_ego=300):
    """
    Create a union graph from multiple ego networks.
    
    Args:
        data_dir: Directory containing the Twitter ego networks
        num_egos: Number of ego networks to include
        edges_per_ego: Maximum number of edges to sample from each ego network
        
    Returns:
        A unified NetworkX undirected graph
    """
    print(f"Creating union graph from up to {num_egos} ego networks with up to {edges_per_ego} edges each...")
    
    # Get list of all available ego networks
    ego_networks = []
    for filename in os.listdir(os.path.join(data_dir, "twitter")):
        if filename.endswith(".edges"):
            ego_networks.append(filename.split(".")[0])
    
    # Randomly sample ego networks if we have more than requested
    if len(ego_networks) > num_egos:
        selected_egos = random.sample(ego_networks, num_egos)
    else:
        selected_egos = ego_networks
        print(f"Warning: Only {len(selected_egos)} ego networks available")
    
    # Create union graph (undirected)
    union_graph = nx.Graph()
    
    # Add edges from each ego network
    for ego_id in selected_egos:
        ego_graph = load_ego_network(ego_id, data_dir)
        
        if ego_graph is None:
            continue
        
        # Sample edges if there are more than the limit
        if ego_graph.number_of_edges() > edges_per_ego:
            edge_list = list(ego_graph.edges())
            sampled_edges = random.sample(edge_list, edges_per_ego)
            ego_graph = nx.Graph()
            ego_graph.add_edges_from(sampled_edges)
        
        # Add all edges to the union graph
        union_graph.add_edges_from(ego_graph.edges())
    
    print(f"Union graph created with {union_graph.number_of_nodes()} nodes and {union_graph.number_of_edges()} edges")
    return union_graph

# ================================
# Implementation 1: Topological Features
# ================================

def extract_topological_features(graph):
    """
    Extract topological features for all nodes in the graph.
    
    Args:
        graph: NetworkX graph
    
    Returns:
        Dictionary mapping nodes to their feature dictionaries
    """
    print("Extracting topological features...")
    
    # Pre-compute node-level metrics to avoid recalculating
    degree_dict = dict(graph.degree())
    
    # For large graphs, compute approximate centrality measures
    if graph.number_of_nodes() > 1000:
        print("  Large graph detected, using approximate centrality measures")
        betweenness_dict = nx.betweenness_centrality(graph, k=500)
        
        # Try eigenvector centrality but fall back to degree centrality if it fails
        try:
            eigenvector_dict = nx.eigenvector_centrality(graph, max_iter=100)
        except:
            print("  Eigenvector centrality calculation failed, using degree centrality instead")
            eigenvector_dict = nx.degree_centrality(graph)
    else:
        betweenness_dict = nx.betweenness_centrality(graph)
        eigenvector_dict = nx.eigenvector_centrality(graph)
    
    closeness_dict = nx.closeness_centrality(graph)
    clustering_dict = nx.clustering(graph)
    
    # Create feature dictionary for each node
    node_features = {}
    for node in graph.nodes():
        node_features[node] = {
            'degree': degree_dict.get(node, 0),
            'betweenness': betweenness_dict.get(node, 0),
            'closeness': closeness_dict.get(node, 0),
            'eigenvector': eigenvector_dict.get(node, 0),
            'clustering': clustering_dict.get(node, 0)
        }
    
    print(f"Extracted topological features for {len(node_features)} nodes")
    return node_features

def compute_edge_features(graph, u, v, node_features):
    """
    Compute features for a pair of nodes based on topological properties.
    
    Args:
        graph: NetworkX graph
        u: First node
        v: Second node
        node_features: Dictionary of pre-computed node features
    
    Returns:
        Dictionary of edge features
    """
    # Skip if either node is not in the graph
    if u not in graph or v not in graph:
        # Return zeros for all features
        return {
            'degree_u': 0, 'degree_v': 0, 'degree_diff': 0,
            'betweenness_u': 0, 'betweenness_v': 0, 'betweenness_diff': 0,
            'closeness_u': 0, 'closeness_v': 0, 'closeness_diff': 0,
            'eigenvector_u': 0, 'eigenvector_v': 0, 'eigenvector_diff': 0,
            'clustering_u': 0, 'clustering_v': 0, 'clustering_diff': 0,
            'common_neighbors': 0, 'jaccard_coefficient': 0,
            'preferential_attachment': 0, 'adamic_adar': 0, 'resource_allocation': 0
        }
    
    # Node-based features
    u_feats = node_features[u]
    v_feats = node_features[v]
    
    features = {
        # Degree-based features
        'degree_u': u_feats['degree'],
        'degree_v': v_feats['degree'],
        'degree_diff': abs(u_feats['degree'] - v_feats['degree']),
        
        # Centrality measures
        'betweenness_u': u_feats['betweenness'],
        'betweenness_v': v_feats['betweenness'],
        'betweenness_diff': abs(u_feats['betweenness'] - v_feats['betweenness']),
        
        'closeness_u': u_feats['closeness'],
        'closeness_v': v_feats['closeness'],
        'closeness_diff': abs(u_feats['closeness'] - v_feats['closeness']),
        
        'eigenvector_u': u_feats['eigenvector'],
        'eigenvector_v': v_feats['eigenvector'],
        'eigenvector_diff': abs(u_feats['eigenvector'] - v_feats['eigenvector']),
        
        'clustering_u': u_feats['clustering'],
        'clustering_v': v_feats['clustering'],
        'clustering_diff': abs(u_feats['clustering'] - v_feats['clustering']),
    }
    
    # Compute pairwise topological features
    
    # Common neighbors - manually compute for compatibility with both directed and undirected graphs
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    common_neighbors = neighbors_u.intersection(neighbors_v)
    common_neighbors_count = len(common_neighbors)
    features['common_neighbors'] = common_neighbors_count
    
    # Jaccard coefficient
    union_size = len(neighbors_u.union(neighbors_v))
    features['jaccard_coefficient'] = common_neighbors_count / union_size if union_size > 0 else 0
    
    # Preferential attachment
    features['preferential_attachment'] = u_feats['degree'] * v_feats['degree']
    
    # Adamic-Adar index
    adamic_adar = 0
    resource_allocation = 0
    for common_neighbor in common_neighbors:
        cn_degree = node_features[common_neighbor]['degree']
        if cn_degree > 1:  # Avoid log(1) = 0 division
            adamic_adar += 1 / np.log(cn_degree)
        resource_allocation += 1 / cn_degree if cn_degree > 0 else 0
    
    features['adamic_adar'] = adamic_adar
    features['resource_allocation'] = resource_allocation
    
    return features

def create_edge_dataset(graph, pos_edges, neg_edges, node_features):
    """
    Create feature matrices and labels for edges.
    
    Args:
        graph: NetworkX graph
        pos_edges: List of positive edges (u, v)
        neg_edges: List of negative edges (u, v)
        node_features: Dictionary of pre-computed node features
    
    Returns:
        X: Feature matrix
        y: Labels (1 for positive edges, 0 for negative edges)
        feature_names: List of feature names
    """
    print("Creating edge feature dataset...")
    
    # Combine positive and negative edges
    all_edges = pos_edges + neg_edges
    y = np.array([1] * len(pos_edges) + [0] * len(neg_edges))
    
    # Extract features for the first edge to get feature names
    first_edge = all_edges[0]
    first_features = compute_edge_features(graph, first_edge[0], first_edge[1], node_features)
    feature_names = list(first_features.keys())
    
    # Initialize feature matrix
    X = np.zeros((len(all_edges), len(feature_names)))
    
    # Fill feature matrix
    for i, (u, v) in enumerate(all_edges):
        features = compute_edge_features(graph, u, v, node_features)
        X[i] = [features[fname] for fname in feature_names]
    
    print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
    return X, y, feature_names

def run_implementation1(union_graph):
    """
    Run Implementation 1 (topological features) on the union graph.
    
    Args:
        union_graph: NetworkX undirected graph of the union of multiple ego networks
        
    Returns:
        Dictionary containing the results
    """
    print("\n========== Running Implementation 1: Topological Features ==========")
    
    # 1. Split edges into training and testing sets
    edges = list(union_graph.edges())
    train_edges, test_edges = train_test_split(edges, test_size=0.3, random_state=42)
    
    # 2. Create a graph with only training edges (for feature extraction)
    train_graph = nx.Graph()
    train_graph.add_nodes_from(union_graph.nodes())
    train_graph.add_edges_from(train_edges)
    
    print(f"Training graph: {train_graph.number_of_nodes()} nodes, {train_graph.number_of_edges()} edges")
    
    # 3. Sample non-edges for testing
    test_non_edges = []
    all_nodes = list(union_graph.nodes())
    
    print("Sampling negative examples...")
    
    while len(test_non_edges) < len(test_edges):
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u != v and not union_graph.has_edge(u, v) and (u, v) not in test_non_edges and (v, u) not in test_non_edges:
            test_non_edges.append((u, v))
    
    # 4. Sample non-edges for training
    train_non_edges = []
    while len(train_non_edges) < len(train_edges):
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if (u != v and not union_graph.has_edge(u, v) and not union_graph.has_edge(v, u) and
            (u, v) not in test_non_edges and (v, u) not in test_non_edges and
            (u, v) not in train_non_edges and (v, u) not in train_non_edges):
            train_non_edges.append((u, v))
    
    print(f"Training set: {len(train_edges)} positive, {len(train_non_edges)} negative")
    print(f"Testing set: {len(test_edges)} positive, {len(test_non_edges)} negative")
    
    # 5. Extract topological features
    node_features = extract_topological_features(train_graph)
    
    # 6. Create feature matrices
    X_train, y_train, feature_names = create_edge_dataset(train_graph, train_edges, train_non_edges, node_features)
    X_test, y_test, _ = create_edge_dataset(train_graph, test_edges, test_non_edges, node_features)
    
    # 7. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 8. Train models with different numbers of trees
    n_estimators_list = [1, 5, 10, 25, 50, 100, 200]
    train_scores = []
    test_scores = []
    oob_scores = []
    
    print("\nTraining Random Forest models with varying numbers of trees:")
    for n_estimators in n_estimators_list:
        print(f"  Training with {n_estimators} trees...")
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1, oob_score=True)
        rf.fit(X_train_scaled, y_train)
        
        # Record scores
        train_scores.append(rf.score(X_train_scaled, y_train))
        test_scores.append(rf.score(X_test_scaled, y_test))
        oob_scores.append(rf.oob_score_)
    
    # 9. Train final model with 200 trees
    print("\nTraining final model with 200 trees...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_train_scaled, y_train)
    
    # 10. Evaluate the model
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults for Implementation 1:")
    print(f"  AUC Score: {auc_score:.4f}")
    print(f"  Average Precision: {ap_score:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # 11. Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # 12. Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    # 13. Calculate scores at different thresholds
    thresholds_to_test = np.linspace(0, 1, 101)[1:-1]  # from 0.01 to 0.99
    precision_at_thresholds = []
    recall_at_thresholds = []
    f1_at_thresholds = []
    accuracy_at_thresholds = []
    
    for threshold in thresholds_to_test:
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        
        # Handle division by zero
        if sum(y_pred_at_threshold) > 0:
            prec = np.sum((y_pred_at_threshold == 1) & (y_test == 1)) / np.sum(y_pred_at_threshold == 1)
        else:
            prec = 0
            
        if sum(y_test) > 0:
            rec = np.sum((y_pred_at_threshold == 1) & (y_test == 1)) / np.sum(y_test == 1)
        else:
            rec = 0
            
        precision_at_thresholds.append(prec)
        recall_at_thresholds.append(rec)
        
        # Calculate F1 score with handling for edge cases
        if prec + rec > 0:
            f1_at_thresholds.append(2 * prec * rec / (prec + rec))
        else:
            f1_at_thresholds.append(0)
            
        accuracy_at_thresholds.append(np.mean(y_pred_at_threshold == y_test))
    
    # 14. Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_scores, 'o-', label='Training Accuracy')
    plt.plot(n_estimators_list, test_scores, 'o-', label='Testing Accuracy')
    plt.plot(n_estimators_list, oob_scores, 'o-', label='Out-of-Bag Accuracy')
    plt.xlabel('Number of Trees in Random Forest')
    plt.ylabel('Accuracy')
    plt.title('Implementation 1: Learning Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("imp1_learning_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 15. Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Implementation 1: Precision-Recall Curve (AP = {ap_score:.4f})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("imp1_precision_recall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 16. Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Implementation 1: ROC Curve (AUC = {auc_score:.4f})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("imp1_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 17. Plot metrics vs. threshold
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds_to_test, precision_at_thresholds, 'r-', label='Precision')
    plt.plot(thresholds_to_test, recall_at_thresholds, 'g-', label='Recall')
    plt.plot(thresholds_to_test, f1_at_thresholds, 'b-', label='F1 Score')
    plt.plot(thresholds_to_test, accuracy_at_thresholds, 'k-', label='Accuracy')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Implementation 1: Metrics vs. Classification Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("imp1_classification_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 18. Plot feature importance
    feature_importance = rf.feature_importances_
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(sorted_features))
    plt.barh(y_pos, sorted_importance, align='center')
    plt.yticks(y_pos, sorted_features)
    plt.xlabel('Feature Importance')
    plt.title('Implementation 1: Feature Importance')
    plt.tight_layout()
    plt.savefig("imp1_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compile results
    results = {
        'auc_score': auc_score,
        'ap_score': ap_score,
        'f1_score': f1,
        'accuracy': accuracy,
        'n_estimators_list': n_estimators_list,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'oob_scores': oob_scores,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds_to_test': thresholds_to_test,
        'precision_at_thresholds': precision_at_thresholds,
        'recall_at_thresholds': recall_at_thresholds,
        'f1_at_thresholds': f1_at_thresholds,
        'accuracy_at_thresholds': accuracy_at_thresholds,
        'feature_importance': dict(zip(feature_names, feature_importance))
    }
    
    return results

# ================================
# Implementation 2: Node2Vec
# ================================

def learn_node2vec_embeddings(graph, dimensions=64, walk_length=30, num_walks=200):
    """
    Learn node embeddings using Node2Vec.
    
    Args:
        graph: NetworkX graph
        dimensions: Embedding dimensions
        walk_length: Length of each random walk
        num_walks: Number of random walks per node
    
    Returns:
        Dictionary mapping node IDs to their embedding vectors
    """
    print(f"Learning {dimensions}-dimensional Node2Vec embeddings...")
    
    # Initialize Node2Vec
    node2vec = Node2Vec(
        graph, 
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=4
    )
    
    # Train the model
    model = node2vec.fit(
        window=10,
        min_count=1,
        batch_words=4
    )
    
    # Create a dictionary mapping node IDs to embeddings
    embeddings = {}
    for node in graph.nodes():
        try:
            # Convert node to string for gensim
            node_str = str(node)
            embeddings[node] = model.wv[node_str]
        except KeyError:
            # If a node wasn't included in any random walks
            embeddings[node] = np.zeros(dimensions)
    
    print(f"Generated embeddings for {len(embeddings)} nodes")
    return embeddings

def run_implementation2(union_graph):
    """
    Run Implementation 2 (Node2Vec) on the union graph.
    
    Args:
        union_graph: NetworkX undirected graph of the union of multiple ego networks
        
    Returns:
        Dictionary containing the results
    """
    print("\n========== Running Implementation 2: Node2Vec ==========")
    
    # 1. Split edges into training and testing sets
    edges = list(union_graph.edges())
    train_edges, test_edges = train_test_split(edges, test_size=0.3, random_state=42)
    
    # 2. Create a graph with only training edges
    train_graph = nx.Graph()
    train_graph.add_nodes_from(union_graph.nodes())
    train_graph.add_edges_from(train_edges)
    
    print(f"Training graph: {train_graph.number_of_nodes()} nodes, {train_graph.number_of_edges()} edges")
    
    # 3. Sample non-edges for testing
    test_non_edges = []
    all_nodes = list(union_graph.nodes())
    
    print("Sampling negative examples...")
    
    while len(test_non_edges) < len(test_edges):
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u != v and not union_graph.has_edge(u, v) and not union_graph.has_edge(v, u) and (u, v) not in test_non_edges and (v, u) not in test_non_edges:
            test_non_edges.append((u, v))
    
    # 4. Sample non-edges for training
    train_non_edges = []
    while len(train_non_edges) < len(train_edges):
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if (u != v and not union_graph.has_edge(u, v) and not union_graph.has_edge(v, u) and
            (u, v) not in test_non_edges and (v, u) not in test_non_edges and
            (u, v) not in train_non_edges and (v, u) not in train_non_edges):
            train_non_edges.append((u, v))
    
    print(f"Training set: {len(train_edges)} positive, {len(train_non_edges)} negative")
    print(f"Testing set: {len(test_edges)} positive, {len(test_non_edges)} negative")
    
    # 5. Learn node embeddings from the training graph
    embeddings = learn_node2vec_embeddings(train_graph, dimensions=64, walk_length=30, num_walks=200)
    
    # 6. Create feature vectors using Hadamard product
    print("Creating edge feature vectors...")
    X_train = []
    y_train = []
    
    # Create training features
    for u, v in train_edges:
        if u in embeddings and v in embeddings:
            edge_embedding = embeddings[u] * embeddings[v]  # Hadamard product
            X_train.append(edge_embedding)
            y_train.append(1)
    
    for u, v in train_non_edges:
        if u in embeddings and v in embeddings:
            edge_embedding = embeddings[u] * embeddings[v]
            X_train.append(edge_embedding)
            y_train.append(0)
    
    # Create testing features
    X_test = []
    y_test = []
    
    for u, v in test_edges:
        if u in embeddings and v in embeddings:
            edge_embedding = embeddings[u] * embeddings[v]
            X_test.append(edge_embedding)
            y_test.append(1)
    
    for u, v in test_non_edges:
        if u in embeddings and v in embeddings:
            edge_embedding = embeddings[u] * embeddings[v]
            X_test.append(edge_embedding)
            y_test.append(0)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Created training set with {X_train.shape[0]} samples and {X_train.shape[1]} features")
    print(f"Created testing set with {X_test.shape[0]} samples and {X_test.shape[1]} features")
    
    # 7. Train models with different numbers of trees
    n_estimators_list = [1, 5, 10, 25, 50, 100, 200]
    train_scores = []
    test_scores = []
    oob_scores = []
    
    print("\nTraining Random Forest models with varying numbers of trees:")
    for n_estimators in n_estimators_list:
        print(f"  Training with {n_estimators} trees...")
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1, oob_score=True)
        rf.fit(X_train, y_train)
        
        # Record scores
        train_scores.append(rf.score(X_train, y_train))
        test_scores.append(rf.score(X_test, y_test))
        oob_scores.append(rf.oob_score_)
    
    # 8. Train final model with 200 trees
    print("\nTraining final model with 200 trees...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_train, y_train)
    
    # 9. Evaluate the model
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults for Implementation 2:")
    print(f"  AUC Score: {auc_score:.4f}")
    print(f"  Average Precision: {ap_score:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # 10. Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # 11. Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    # 12. Calculate scores at different thresholds
    thresholds_to_test = np.linspace(0, 1, 101)[1:-1]  # from 0.01 to 0.99
    precision_at_thresholds = []
    recall_at_thresholds = []
    f1_at_thresholds = []
    accuracy_at_thresholds = []
    
    for threshold in thresholds_to_test:
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        
        # Handle division by zero
        if sum(y_pred_at_threshold) > 0:
            prec = np.sum((y_pred_at_threshold == 1) & (y_test == 1)) / np.sum(y_pred_at_threshold == 1)
        else:
            prec = 0
            
        if sum(y_test) > 0:
            rec = np.sum((y_pred_at_threshold == 1) & (y_test == 1)) / np.sum(y_test == 1)
        else:
            rec = 0
            
        precision_at_thresholds.append(prec)
        recall_at_thresholds.append(rec)
        
        # Calculate F1 score with handling for edge cases
        if prec + rec > 0:
            f1_at_thresholds.append(2 * prec * rec / (prec + rec))
        else:
            f1_at_thresholds.append(0)
            
        accuracy_at_thresholds.append(np.mean(y_pred_at_threshold == y_test))
    
    # 13. Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_scores, 'o-', label='Training Accuracy')
    plt.plot(n_estimators_list, test_scores, 'o-', label='Testing Accuracy')
    plt.plot(n_estimators_list, oob_scores, 'o-', label='Out-of-Bag Accuracy')
    plt.xlabel('Number of Trees in Random Forest')
    plt.ylabel('Accuracy')
    plt.title('Implementation 2: Learning Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("imp2_learning_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 14. Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Implementation 2: Precision-Recall Curve (AP = {ap_score:.4f})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("imp2_precision_recall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 15. Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Implementation 2: ROC Curve (AUC = {auc_score:.4f})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("imp2_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 16. Plot metrics vs. threshold
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds_to_test, precision_at_thresholds, 'r-', label='Precision')
    plt.plot(thresholds_to_test, recall_at_thresholds, 'g-', label='Recall')
    plt.plot(thresholds_to_test, f1_at_thresholds, 'b-', label='F1 Score')
    plt.plot(thresholds_to_test, accuracy_at_thresholds, 'k-', label='Accuracy')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Implementation 2: Metrics vs. Classification Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("imp2_classification_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 17. Get feature importance for top dimensions
    feature_importance = rf.feature_importances_
    
    # Compile results
    results = {
        'auc_score': auc_score,
        'ap_score': ap_score,
        'f1_score': f1,
        'accuracy': accuracy,
        'n_estimators_list': n_estimators_list,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'oob_scores': oob_scores,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds_to_test': thresholds_to_test,
        'precision_at_thresholds': precision_at_thresholds,
        'recall_at_thresholds': recall_at_thresholds,
        'f1_at_thresholds': f1_at_thresholds,
        'accuracy_at_thresholds': accuracy_at_thresholds,
        'feature_importance': feature_importance
    }
    
    return results

# ================================
# Comparison Function
# ================================

def compare_implementations(imp1_results, imp2_results):
    """
    Compare results from both implementations and create visualizations.
    
    Args:
        imp1_results: Results dictionary from Implementation 1
        imp2_results: Results dictionary from Implementation 2
    """
    print("\n========== Comparing Implementation 1 and Implementation 2 ==========")
    
    # Create a results table
    print("\nPerformance Comparison:")
    print(f"{'Metric':<20} {'Implementation 1':<20} {'Implementation 2':<20} {'Difference':<20}")
    print("-" * 80)
    
    metrics = [
        ('AUC', imp1_results['auc_score'], imp2_results['auc_score']),
        ('Average Precision', imp1_results['ap_score'], imp2_results['ap_score']),
        ('F1 Score', imp1_results['f1_score'], imp2_results['f1_score']),
        ('Accuracy', imp1_results['accuracy'], imp2_results['accuracy'])
    ]
    
    for metric_name, imp1_value, imp2_value in metrics:
        diff = imp2_value - imp1_value
        diff_pct = diff / imp1_value * 100 if imp1_value > 0 else float('inf')
        
        print(f"{metric_name:<20} {imp1_value:.4f}{'':<14} {imp2_value:.4f}{'':<14} {diff:.4f} ({diff_pct:+.2f}%)")
    
    # 1. AUC and AP comparison bar chart
    plt.figure(figsize=(10, 6))
    
    metrics = ['AUC', 'AP']
    imp1_values = [imp1_results['auc_score'], imp1_results['ap_score']]
    imp2_values = [imp2_results['auc_score'], imp2_results['ap_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, imp1_values, width, label='Implementation 1 (Topological)')
    plt.bar(x + width/2, imp2_values, width, label='Implementation 2 (Node2Vec)')
    
    plt.ylabel('Score')
    plt.title('Performance Comparison: Implementation 1 vs. Implementation 2')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comparison_auc_ap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1 and Accuracy comparison bar chart
    plt.figure(figsize=(10, 6))
    
    metrics = ['F1', 'Accuracy']
    imp1_values = [imp1_results['f1_score'], imp1_results['accuracy']]
    imp2_values = [imp2_results['f1_score'], imp2_results['accuracy']]
    
    x = np.arange(len(metrics))
    
    plt.bar(x - width/2, imp1_values, width, label='Implementation 1 (Topological)')
    plt.bar(x + width/2, imp2_values, width, label='Implementation 2 (Node2Vec)')
    
    plt.ylabel('Score')
    plt.title('Performance Comparison: Implementation 1 vs. Implementation 2')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comparison_f1_acc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC curve comparison
    plt.figure(figsize=(10, 6))
    plt.plot(imp1_results['fpr'], imp1_results['tpr'], lw=2, 
             label=f'Implementation 1 (AUC = {imp1_results["auc_score"]:.4f})')
    plt.plot(imp2_results['fpr'], imp2_results['tpr'], lw=2, 
             label=f'Implementation 2 (AUC = {imp2_results["auc_score"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig("comparison_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall curve comparison
    plt.figure(figsize=(10, 6))
    plt.plot(imp1_results['recall'], imp1_results['precision'], lw=2, 
             label=f'Implementation 1 (AP = {imp1_results["ap_score"]:.4f})')
    plt.plot(imp2_results['recall'], imp2_results['precision'], lw=2, 
             label=f'Implementation 2 (AP = {imp2_results["ap_score"]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig("comparison_precision_recall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Learning curve comparison
    plt.figure(figsize=(10, 6))
    plt.plot(imp1_results['n_estimators_list'], imp1_results['test_scores'], 'o-', 
             label='Implementation 1 Testing Accuracy')
    plt.plot(imp2_results['n_estimators_list'], imp2_results['test_scores'], 'o-', 
             label='Implementation 2 Testing Accuracy')
    plt.xlabel('Number of Trees in Random Forest')
    plt.ylabel('Testing Accuracy')
    plt.title('Learning Curve Comparison')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("comparison_learning_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plots saved:")
    print("- comparison_auc_ap.png - AUC and Average Precision comparison")
    print("- comparison_f1_acc.png - F1 and Accuracy comparison")
    print("- comparison_roc.png - ROC curve comparison")
    print("- comparison_precision_recall.png - Precision-Recall curve comparison")
    print("- comparison_learning_curve.png - Learning curve comparison")

# ================================
# Main Function
# ================================

def main():
    """
    Main function to run both implementations and compare them.
    """
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare Implementation 1 and Implementation 2')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the dataset')
    parser.add_argument('--num_egos', type=int, default=50, help='Number of ego networks to use')
    parser.add_argument('--edges_per_ego', type=int, default=300, help='Max edges per ego network')
    parser.add_argument('--skip_imp1', action='store_true', help='Skip Implementation 1')
    parser.add_argument('--skip_imp2', action='store_true', help='Skip Implementation 2')
    parser.add_argument('--load_results', action='store_true', help='Load previously saved results')
    
    args = parser.parse_args()
    
    print("======================================================")
    print("   Link Prediction: Implementation 1 vs Implementation 2")
    print("======================================================")
    print(f"Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Number of ego networks: {args.num_egos}")
    print(f"  Edges per ego network: {args.edges_per_ego}")
    print("======================================================")
    
    # Check if we should load previous results
    imp1_results = None
    imp2_results = None
    
    if args.load_results:
        try:
            print("Loading previously saved results...")
            import pickle
            with open('imp1_results.pkl', 'rb') as f:
                imp1_results = pickle.load(f)
            with open('imp2_results.pkl', 'rb') as f:
                imp2_results = pickle.load(f)
            print("Results loaded successfully.")
        except:
            print("Failed to load results, will run implementations.")
            args.load_results = False
            
    # Create the union graph
    if not args.load_results:
        # Create the multi-ego union graph
        start_time = time.time()
        union_graph = create_union_graph(args.data_dir, args.num_egos, args.edges_per_ego)
        print(f"Union graph creation time: {(time.time() - start_time):.2f} seconds")
    
    # Run Implementation 1
    if not args.skip_imp1 and imp1_results is None:
        start_time = time.time()
        imp1_results = run_implementation1(union_graph)
        print(f"Implementation 1 total runtime: {(time.time() - start_time):.2f} seconds")
        
        # Save results
        import pickle
        with open('imp1_results.pkl', 'wb') as f:
            pickle.dump(imp1_results, f)
    
    # Run Implementation 2
    if not args.skip_imp2 and imp2_results is None:
        start_time = time.time()
        imp2_results = run_implementation2(union_graph)
        print(f"Implementation 2 total runtime: {(time.time() - start_time):.2f} seconds")
        
        # Save results
        import pickle
        with open('imp2_results.pkl', 'wb') as f:
            pickle.dump(imp2_results, f)
    
    # Compare implementations
    if imp1_results is not None and imp2_results is not None:
        compare_implementations(imp1_results, imp2_results)
    else:
        if args.skip_imp1:
            print("Implementation 1 was skipped, cannot compare.")
        if args.skip_imp2:
            print("Implementation 2 was skipped, cannot compare.")

if __name__ == "__main__":
    main()