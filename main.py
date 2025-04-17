import matplotlib.pyplot as plt
import os
from load import download_data
from predict import run_link_prediction

if __name__ == "__main__":
    # URL of the dataset
    url = "https://snap.stanford.edu/data/twitter.tar.gz"
    data_dir = "./data"

    # Download and extract the dataset
    download_data(url, data_dir)

    # List available ego networks
    ego_networks = []
    for filename in os.listdir(os.path.join(data_dir, "twitter")):
        if filename.endswith(".edges"):
            ego_networks.append(filename.split(".")[0])

    print(f"Found {len(ego_networks)} ego networks: {ego_networks}")

    # Select a single ego network to analyze
    if len(ego_networks) > 0:
        # You can change this index to select a different network
        network_index = 0
        ego_id = ego_networks[network_index]

        try:
            # Run link prediction
            results = run_link_prediction(ego_id, data_dir)

            # Print a summary of the results
            print("\nLink Prediction Results Summary:")
            print(f"ROC AUC Score: {results['auc_score']:.4f}")
            print(f"Average Precision Score: {results['ap_score']:.4f}")

            print("\nTop 5 Most Important Features:")
            sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"  - {feature}: {importance:.4f}")

            # Plot training and testing accuracy vs. number of trees
            plt.figure(figsize=(10, 6))
            plt.plot(results['n_estimators_list'], results['train_scores'], 'o-', label='Training Accuracy')
            plt.plot(results['n_estimators_list'], results['test_scores'], 'o-', label='Testing Accuracy')
            plt.plot(results['n_estimators_list'], results['oob_scores'], 'o-', label='Out-of-Bag Accuracy')

            plt.xlabel('Number of Trees in Random Forest')
            plt.ylabel('Accuracy')
            plt.title('Learning Curve: Model Performance vs. Ensemble Size')
            plt.grid(alpha=0.3)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"learning_curve_ego_{ego_id}.png", dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print("\nAn error occurred during link prediction:")
            print(str(e))
            import traceback
            traceback.print_exc()
    else:
        print("No ego networks found. Check the dataset extraction.")