#!/usr/bin/env python3
"""
Analyze and cluster article embeddings using K-means and t-SNE visualization.
This script visualizes clusters of similar articles based on their embeddings.
"""

import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from generate_embeddings import Article, Embedding  # Import our models


class EmbeddingAnalyzer:
    def __init__(self, db_engine):
        self.db_engine = db_engine

    def get_all_embeddings(self):
        """Retrieve all embeddings from database"""
        Session = sessionmaker(bind=self.db_engine)
        session = Session()

        results = session.query(Article, Embedding).join(
            Embedding
        ).filter(
            Embedding.embedding_type == 'combined'
        ).all()

        embeddings = []
        titles = []
        categories = []

        for article, embedding in results:
            vector = np.array(json.loads(embedding.vector_json))
            embeddings.append(vector)
            titles.append(article.title)
            categories.append(article.category)

        session.close()

        if not embeddings:
            raise ValueError("No embeddings found in the database!")

        return np.array(embeddings), titles, categories

    def cluster_articles(self, n_clusters=5):
        """Cluster articles using K-means"""
        embeddings, titles, categories = self.get_all_embeddings()

        # Perform clustering
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Visualize with t-SNE
        print("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Article Clusters')

        # Save plot
        plt.savefig('article_clusters.png')
        print("Saved cluster visualization to article_clusters.png")

        # Print cluster information
        for cluster_id in range(n_clusters):
            cluster_indices = clusters == cluster_id
            print(f"\nCluster {cluster_id}:")
            print(f"Number of articles: {sum(cluster_indices)}")
            print("Sample titles:")
            for title in np.array(titles)[cluster_indices][:5]:
                print(f"- {title}")

        return clusters, titles, categories


def main():
    # Initialize database connection
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "coindesk_embeddings.db")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    print(f"Connecting to database at {db_path}")
    sqlite_engine = create_engine(f"sqlite:///{db_path}")

    try:
        # Create analyzer and perform clustering
        analyzer = EmbeddingAnalyzer(sqlite_engine)
        print("\nAnalyzing article clusters...")
        clusters, titles, categories = analyzer.cluster_articles()

        # Additional analysis: Category distribution in clusters
        print("\nCategory distribution in clusters:")
        unique_categories = set(categories)
        for cluster_id in range(len(set(clusters))):
            cluster_indices = clusters == cluster_id
            cluster_categories = [cat for i, cat in enumerate(
                categories) if cluster_indices[i]]
            print(f"\nCluster {cluster_id} category distribution:")
            for category in unique_categories:
                count = cluster_categories.count(category)
                if count > 0:
                    print(f"- {category}: {count}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
