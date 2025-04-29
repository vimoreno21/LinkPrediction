import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from sklearn.tree import export_graphviz
import graphviz
from utils import extract_topological_features, prepare_link_prediction_data, create_edge_dataset, compute_edge_features_optimized, compute_edge_features
import graphviz

from load import load_network, load_features
from evaluate import evaluate_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import numpy as np, matplotlib.pyplot as plt, graphviz

# ---------- helper ----------
def sample_negatives(G, k, rng):
    nodes = list(G.nodes())
    neg = set()
    while len(neg) < k:
        u, v = rng.choice(nodes), rng.choice(nodes)
        if u != v and not G.has_edge(u, v):
            neg.add((u, v))
    return list(neg)

# ---------- main ----------
def run_link_prediction_on_merged_graph(G_train,
                                        test_pos,
                                        test_neg,
                                        neg_ratio_train=1,
                                        rng=np.random.default_rng(42)):
    """
    G_train : graph with *no* held-out edges
    test_pos / test_neg : lists of (u,v) tuples
    """

    # 1) features from leak-free graph
    topo = extract_topological_features(G_train)

    # 2) build training edge lists
    train_pos = list(G_train.edges())
    train_neg = sample_negatives(G_train,
                                 k=len(train_pos) * neg_ratio_train,
                                 rng=rng)

    # 3) vectorise
    X_tr, y_tr, f_names = create_edge_dataset(G_train, train_pos, train_neg, topo)
    X_te, y_te, _        = create_edge_dataset(G_train, test_pos,  test_neg,  topo)

    # 4) scale + fit
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=100,
                                random_state=42,
                                oob_score=True,
                                n_jobs=-1)
    rf.fit(X_tr, y_tr)

    # 5) evaluate
    y_prob = rf.predict_proba(X_te)[:, 1]
    auc, ap, prec, rec, f1, acc = evaluate_model(y_te, y_prob)

    # 6) quick feature importance plot
    fi = rf.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(f_names)[np.argsort(fi)], np.sort(fi))
    plt.title('Feature importance'); plt.tight_layout()
    plt.savefig('feature_importance_merged.png', dpi=300); plt.close()

    # optional: export one tree
    dot = export_graphviz(rf.estimators_[0], feature_names=f_names,
                          class_names=["No-link", "Link"],
                          filled=True, rounded=True, max_depth=3)
    graphviz.Source(dot).render('sample_tree_merged')

    return dict(auc_score=auc, ap_score=ap, precision=prec,
                recall=rec, f1=f1, accuracy=acc,
                feature_importance=dict(zip(f_names, fi)))



def run_link_prediction(ego_id, data_dir):
    """
    Run the complete link prediction pipeline for a given ego network

    This is the main function that orchestrates the entire machine learning workflow:
    1. Data loading and preprocessing
    2. Feature extraction
    3. Train/test splitting
    4. Model training and evaluation
    5. Visualization of results

    Args:
        ego_id: ID of the ego network to analyze
        data_dir: Directory containing the dataset

    Returns:
        Dictionary containing performance metrics and feature importance
    """
    print(f"\n======= Running Link Prediction for Ego Network {ego_id} =======")

    # ---------- STEP 1: DATA LOADING ----------

    print("\n[1/7] Loading network data...")
    # Load the network structure (nodes and edges)
    G, circles = load_network(ego_id, data_dir)

    # Load profile features if available (not used in this implementation)
    node_attrs, feature_names = load_features(ego_id, data_dir)

    print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if circles:
        print(f"Network contains {len(circles)} circles (communities)")

    # ---------- STEP 2: FEATURE EXTRACTION ----------

    print("\n[2/7] Extracting topological features for each node...")
    # Calculate network metrics for each node
    topo_features = extract_topological_features(G)
    print(f"Extracted features for {len(topo_features)} nodes")

    # ---------- STEP 3: DATA SPLITTING ----------

    print("\n[3/7] Splitting data into training and testing sets...")
    # Split existing edges into train/test and sample non-edges
    G_train, train_edges, test_edges, train_non_edges, test_non_edges = prepare_link_prediction_data(G)

    # Data splitting summary
    print(f"\nData splitting summary:")
    print(f"  - Full network: {G.number_of_edges()} edges")
    print(f"  - Training network: {G_train.number_of_edges()} edges ({G_train.number_of_edges()/G.number_of_edges()*100:.1f}%)")
    print(f"  - Training set: {len(train_edges)} positive edges, {len(train_non_edges)} negative edges")
    print(f"  - Testing set: {len(test_edges)} positive edges, {len(test_non_edges)} negative edges")

    # ---------- STEP 4: FEATURE VECTOR CREATION ----------

    print("\n[4/7] Creating feature vectors for edges...")
    # Create feature matrices for training and testing
    X_train, y_train = create_edge_dataset(G_train, train_edges, train_non_edges, topo_features)
    X_test, y_test = create_edge_dataset(G_train, test_edges, test_non_edges, topo_features)

    print(f"Created {len(X_train)} training samples and {len(X_test)} testing samples")
    print(f"Each edge has {X_train.shape[1]} features")

    # ---------- STEP 5: FEATURE SCALING ----------

    print("\n[5/7] Scaling features...")
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- STEP 6: MODEL TRAINING ----------

    print("\n[6/7] Training Random Forest classifier...")

    # --- 6.1: Analyze performance across different numbers of trees ---
    print("  Analyzing performance with different numbers of trees...")

    # Number of trees to try
    n_estimators_list = [1, 5, 10, 25, 50, 75, 100, 150, 200]

    # Store results for plotting
    train_scores = []
    test_scores = []
    oob_scores = []
    training_times = []

    for n_trees in n_estimators_list:
        start_time = time.time()

        # Train Random Forest with current number of trees
        # Include out-of-bag score calculation
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    oob_score=True,
                                    n_jobs=-1,  # Use all available processors
                                    random_state=42)

        rf.fit(X_train_scaled, y_train)

        # Record training time
        training_time = time.time() - start_time
        training_times.append(training_time)

        # Calculate scores
        train_score = rf.score(X_train_scaled, y_train)
        test_score = rf.score(X_test_scaled, y_test)
        oob_score = rf.oob_score_

        # Store scores
        train_scores.append(train_score)
        test_scores.append(test_score)
        oob_scores.append(oob_score)

        print(f"    Trees: {n_trees}, Train: {train_score:.4f}, Test: {test_score:.4f}, OOB: {oob_score:.4f}, Time: {training_time:.2f}s")

    # --- 6.2: Visualize performance vs. number of trees ---
    plt.figure(figsize=(12, 8))

    # Plot accuracy curves
    plt.subplot(2, 1, 1)
    plt.plot(n_estimators_list, train_scores, 'o-', label='Training Accuracy')
    plt.plot(n_estimators_list, test_scores, 'o-', label='Testing Accuracy')
    plt.plot(n_estimators_list, oob_scores, 'o-', label='Out-of-Bag Accuracy')

    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Performance vs. Number of Trees')
    plt.grid(alpha=0.3)
    plt.legend()

    # Plot training time
    plt.subplot(2, 1, 2)
    plt.plot(n_estimators_list, training_times, 'o-', color='red')
    plt.xlabel('Number of Trees')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. Number of Trees')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"rf_trees_performance_ego_{ego_id}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- 6.3: Analyze feature importance stability across different forests ---
    print("  Analyzing feature importance stability...")

    # Train multiple forests and track feature importance
    n_iterations = 10
    feature_names = list(compute_edge_features(G, list(G.nodes())[0], list(G.nodes())[1], topo_features).keys())
    importance_matrix = np.zeros((n_iterations, len(feature_names)))

    for i in range(n_iterations):
        # Train model with different random seed
        rf = RandomForestClassifier(n_estimators=100, random_state=i)
        rf.fit(X_train_scaled, y_train)

        # Store feature importance
        importance_matrix[i, :] = rf.feature_importances_

    # Calculate mean and standard deviation of importance
    mean_importance = np.mean(importance_matrix, axis=0)
    std_importance = np.std(importance_matrix, axis=0)

    # Sort features by mean importance
    sorted_idx = np.argsort(mean_importance)
    sorted_feature_names = [feature_names[i] for i in sorted_idx]

    # Plot feature importance with error bars
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(sorted_feature_names))

    plt.barh(y_pos, mean_importance[sorted_idx], xerr=std_importance[sorted_idx],
             align='center', alpha=0.7, capsize=5)
    plt.yticks(y_pos, sorted_feature_names)
    plt.xlabel('Mean Feature Importance')
    plt.title('Feature Importance Stability Across Multiple Random Forests')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"feature_importance_stability_ego_{ego_id}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- 6.4: Train final model with optimal parameters ---
    print("  Training final model with 100 trees...")

    # Train the final model
    final_rf = RandomForestClassifier(
        n_estimators=75,      # Number of trees in the forest
        max_depth=None,        # Maximum depth of the trees
        min_samples_split=2,   # Minimum samples required to split an internal node
        min_samples_leaf=1,    # Minimum samples required at a leaf node
        random_state=42        # Random seed for reproducibility
    )

    # Fit the model on the training data
    final_rf.fit(X_train_scaled, y_train)

    # Generate predictions (probability scores)
    y_pred_proba = final_rf.predict_proba(X_test_scaled)[:, 1]

    # --- 6.5: Visualize a sample tree from the forest ---
    print("  Generating visualization of a sample decision tree...")

    # Get a single tree from the forest
    estimator = final_rf.estimators_[0]



    # Export tree to DOT format
    dot_data = export_graphviz(
        estimator,
        out_file=None,
        feature_names=feature_names,
        class_names=['No Link', 'Link'],
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3  # Limit depth for visualization
    )

    # Save the tree visualization
    graph = graphviz.Source(dot_data)
    graph.render(f"sample_tree_ego_{ego_id}")

    # ---------- STEP 7: MODEL EVALUATION ----------

    print("\n[7/7] Evaluating model performance...")
    # Calculate performance metrics and create visualizations
    auc_score, ap_score = evaluate_model(y_test, y_pred_proba)

    # Get feature importance from final model
    feature_importance = final_rf.feature_importances_

    # Return results
    return {
        'auc_score': auc_score,
        'ap_score': ap_score,
        'feature_importance': dict(zip(feature_names, feature_importance)),
        'train_scores': train_scores,
        'test_scores': test_scores,
        'oob_scores': oob_scores,
        'n_estimators_list': n_estimators_list
    }

