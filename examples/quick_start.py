"""
Quick start example for semantic test case clustering.
"""
import numpy as np
from src.data.dataloader import load_demo_data
from src.models.model_factory import ModelFactory
from src.clustering.clustering_engine import ClusteringEngine
from src.clustering.dimensionality_reduction import DimensionalityReducer
from src.visualization.static_plots import plot_cluster_scatter

def main():
    """Quick start demo."""
    print("🔍 Semantic Test Case Clustering - Quick Start")
    print("=" * 50)

    # Step 1: Load data
    print("\n1. Loading demo data...")
    df = load_demo_data(max_samples=200)
    texts = df['test_description'].tolist()
    print(f"   Loaded {len(texts)} test cases")

    # Step 2: Generate embeddings
    print("\n2. Generating BERT embeddings...")
    config = {'name': 'bert', 'embedding_dim': 384}
    embedder = ModelFactory.create_embedder('bert', config)
    embedder.initialize()
    embeddings = embedder.embed_texts(texts)
    print(f"   Generated embeddings: {embeddings.shape}")

    # Step 3: Apply clustering
    print("\n3. Applying K-means clustering...")
    clustering_engine = ClusteringEngine()
    labels = clustering_engine.apply_kmeans(embeddings, n_clusters=6)
    stats = clustering_engine.get_cluster_statistics(embeddings)
    print(f"   Found {stats['n_clusters']} clusters")
    print(f"   Silhouette score: {stats.get('silhouette_score', 0):.3f}")

    # Step 4: Visualize results
    print("\n4. Creating visualization...")
    reducer = DimensionalityReducer()
    coords_2d = reducer.apply_tsne(embeddings, n_components=2)

    plot_cluster_scatter(coords_2d, labels, 
                        title="Quick Start: Test Case Clusters",
                        save_path="outputs/quick_start_clusters.png")

    print("\n✅ Quick start complete!")
    print("   Check 'outputs/quick_start_clusters.png' for visualization")

    # Show sample clusters
    print("\n📋 Sample clusters:")
    for cluster_id in range(min(3, stats['n_clusters'])):
        cluster_indices = np.where(labels == cluster_id)[0][:3]
        print(f"\n   Cluster {cluster_id}:")
        for idx in cluster_indices:
            print(f"   - {texts[idx]}")

if __name__ == "__main__":
    main()
