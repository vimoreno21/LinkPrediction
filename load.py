import os
import tarfile
import urllib.request
import networkx as nx


def download_data(url, extract_dir):
    """
    Download and extract the dataset if it doesn't exist already
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Download the file
    tar_file_path = os.path.join(extract_dir, "ego-twitter.tar.gz")
    if not os.path.exists(tar_file_path):
        print(f"Downloading the dataset from {url}...")
        urllib.request.urlretrieve(url, tar_file_path)

    # Extract the file
    if not os.path.exists(os.path.join(extract_dir, "twitter")):
        print("Extracting the dataset...")
        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(extract_dir)

    print("Data download and extraction complete.")

def load_network(ego_id, data_dir):
    """
    Load the network data for a specific ego ID
    """
    # Path to edge file
    edge_file = os.path.join(data_dir, "twitter", f"{ego_id}.edges")

    # Create a directed graph (since Twitter follows are directed)
    G = nx.DiGraph()

    # Add edges from the edge file
    with open(edge_file, 'r') as f:
        for line in f:
            source, target = map(int, line.strip().split())
            G.add_edge(source, target)

    # Add the ego node (assumes ego follows everyone)
    ego_node = int(ego_id)
    # Create a copy of nodes before modifying the graph
    nodes = list(G.nodes())
    for node in nodes:
        G.add_edge(ego_node, node)

    # Load circles if available
    circles = {}
    circles_file = os.path.join(data_dir, "twitter", f"{ego_id}.circles")
    if os.path.exists(circles_file):
        with open(circles_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                circle_name = parts[0]
                circle_members = [int(x) for x in parts[1:]]
                circles[circle_name] = circle_members

    return G, circles

def load_features(ego_id, data_dir):
    """
    Load node features if available
    """
    features = {}
    feat_file = os.path.join(data_dir, "twitter", f"{ego_id}.feat")

    if os.path.exists(feat_file):
        # Load feature names if available
        feature_names = []
        featnames_file = os.path.join(data_dir, "twitter", f"{ego_id}.featnames")
        if os.path.exists(featnames_file):
            with open(featnames_file, 'r') as f:
                for line in f:
                    feature_names.append(line.strip().split()[1])

        # Load features
        with open(feat_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                node_features = [int(x) for x in parts[1:]]
                features[node_id] = node_features

    return features, feature_names