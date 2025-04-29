import networkx as nx
from sklearn.model_selection import train_test_split
import random
import numpy as np
from evaluate import compute_edge_features_optimized, compute_edge_features
from joblib import Parallel, delayed


def create_edge_dataset(G, pos_edges, neg_edges, node_features):
    G_undirected = G.to_undirected()
    degrees = dict(G_undirected.degree())
    neighbors = {n: set(G_undirected.neighbors(n)) for n in G.nodes()}
    preds = {n: set(G.predecessors(n)) for n in G.nodes()}
    succs = {n: set(G.successors(n)) for n in G.nodes()}

    def process(u, v, label):
        feats = compute_edge_features_optimized(G, G_undirected, degrees, neighbors, preds, succs, u, v, node_features)
        return list(feats.values()), label, list(feats.keys())

    all_samples = [(u, v, 1) for u, v in pos_edges] + [(u, v, 0) for u, v in neg_edges]
    results = Parallel(n_jobs=-1, backend="threading")(delayed(process)(u, v, label) for u, v, label in all_samples)


    X, y, feature_keys = zip(*results)
    return np.array(X), np.array(y), feature_keys[0]  # all keys are the same



def prepare_link_prediction_data(G, test_ratio=0.3, neg_ratio=1.0):
    """
    Prepare data for link prediction with faster negative sampling
    """
    # Extract all edges
    all_edges = list(G.edges())

    # Split edges into training and testing sets
    train_edges, test_edges = train_test_split(all_edges, test_size=test_ratio, random_state=42)

    # Create a new graph with only training edges
    G_train = G.copy()
    G_train.remove_edges_from(test_edges)

    # Get all possible negative edges
    all_non_edges = list(nx.non_edges(G))

    # Shuffle and sample negative edges
    np.random.seed(42)
    np.random.shuffle(all_non_edges)

    n_pos_train = len(train_edges)
    n_pos_test = len(test_edges)

    total_needed = int((n_pos_train + n_pos_test) * neg_ratio)
    sampled_non_edges = all_non_edges[:total_needed]

    train_non_edges = sampled_non_edges[:int(n_pos_train * neg_ratio)]
    test_non_edges = sampled_non_edges[int(n_pos_train * neg_ratio):]

    return G_train, train_edges, test_edges, train_non_edges, test_non_edges

def extract_topological_features(G):
    """
    Extract topological features for each node
    """
    # Get basic centrality measures
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    

    # PageRank as a measure of influence
    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    except:
        pagerank = {node: 0.0 for node in G.nodes()}

    # Local clustering coefficient
    try:
        clustering = nx.clustering(G.to_undirected())
    except:
        clustering = {node: 0.0 for node in G.nodes()}

    # Combine all features
    node_features = {}
    for node in G.nodes():
        node_features[node] = {
            'degree_centrality': degree_centrality.get(node, 0.0),
            'in_degree_centrality': in_degree_centrality.get(node, 0.0),
            'out_degree_centrality': out_degree_centrality.get(node, 0.0),
            'pagerank': pagerank.get(node, 0.0),
            'clustering': clustering.get(node, 0.0)
        }

    return node_features