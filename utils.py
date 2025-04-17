import networkx as nx
from sklearn.model_selection import train_test_split
import random
import numpy as np
from evaluate import compute_edge_features


def create_edge_dataset(G, pos_edges, neg_edges, node_features):
    """
    Create a dataset for edge prediction
    """
    X = []
    y = []

    # Add positive examples
    for u, v in pos_edges:
        edge_feats = compute_edge_features(G, u, v, node_features)
        X.append(list(edge_feats.values()))
        y.append(1)  # Positive class

    # Add negative examples
    for u, v in neg_edges:
        edge_feats = compute_edge_features(G, u, v, node_features)
        X.append(list(edge_feats.values()))
        y.append(0)  # Negative class

    return np.array(X), np.array(y)

def prepare_link_prediction_data(G, test_ratio=0.3, neg_ratio=1.0):
    """
    Prepare data for link prediction
    """
    # Extract all edges
    all_edges = list(G.edges())

    # Split edges into training and testing sets
    train_edges, test_edges = train_test_split(all_edges, test_size=test_ratio, random_state=42)

    # Create a new graph with only training edges
    G_train = G.copy()
    G_train.remove_edges_from(test_edges)

    # Calculate negative edges (non-existing edges)
    nodes = list(G.nodes())

    # Function to sample negative edges
    def sample_non_edges(graph, num_samples, exclude_edges=None):
        if exclude_edges is None:
            exclude_edges = set()
        else:
            exclude_edges = set(exclude_edges)

        non_edges = []
        while len(non_edges) < num_samples:
            # Sample random node pairs
            u, v = random.sample(nodes, 2)
            # Check if this edge exists or is in excluded set
            if u != v and not graph.has_edge(u, v) and (u, v) not in exclude_edges:
                non_edges.append((u, v))
                exclude_edges.add((u, v))

        return non_edges

    # Sample negative edges for training and testing
    n_pos_train = len(train_edges)
    n_pos_test = len(test_edges)

    train_non_edges = sample_non_edges(G, int(n_pos_train * neg_ratio))
    test_non_edges = sample_non_edges(G, int(n_pos_test * neg_ratio), exclude_edges=train_non_edges)

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