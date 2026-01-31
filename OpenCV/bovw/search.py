"""
Similarity Search for Bag of Visual Words

This module provides methods to:
1. Find the most similar images to a query image
2. Determine if a new image belongs to the same group
3. Visualize similarity results
"""

from enum import IntEnum
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
from bag_of_visual_words import BagOfVisualWords


class BoVWMetric(IntEnum):
    COSINE = 0
    EUCLIDEAN = 1
    CHI_SQUARE = 2
    HISTOGRAM_INTERSECTION = 3

    def compare(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if self == BoVWMetric.COSINE:
            """
            Compute cosine similarity between two vectors

            Cosine similarity = (A · B) / (||A|| * ||B||)
            Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite

            For normalized vectors (which BoVW produces), this simplifies to dot product.
            """
            # If vectors are already L2 normalized, cosine similarity is just dot product
            return np.dot(vec1, vec2)

        elif self == BoVWMetric.EUCLIDEAN:
            """
            Compute Euclidean distance between two vectors

            Lower distance = more similar
            Range: [0, ∞)
            """
            return float(np.linalg.norm(vec1 - vec2))

        elif self == BoVWMetric.CHI_SQUARE:
            """
            Compute Chi-square distance between two histograms

            Chi-square distance = 0.5 * Σ((A[i] - B[i])² / (A[i] + B[i]))

            Good for histogram comparison, handles different scales well.
            Lower distance = more similar
            Range: [0, ∞)
            """
            # Avoid division by zero
            epsilon = 1e-10
            return 0.5 * np.sum((vec1 - vec2) ** 2 / (vec1 + vec2 + epsilon))

        elif self == BoVWMetric.HISTOGRAM_INTERSECTION:
            """
            Compute histogram intersection

            Intersection = Σ min(A[i], B[i])

            Higher value = more similar
            Range: [0, 1] for normalized histograms
            """
            return np.sum(np.minimum(vec1, vec2))

        else:
            raise ValueError("Unsupported Metric")


class BoVWSimilaritySearch:
    """
    Handles similarity search and image matching using BoVW features.

    How it works:
    1. Each image is represented as a histogram (feature vector) of visual words
    2. Similarity is measured by comparing these histograms
    3. Common metrics: Cosine similarity, Euclidean distance, Chi-square distance
    """

    def __init__(self, bovw_model: BagOfVisualWords):
        """
        Initialize similarity search

        Args:
            bovw_model: Trained BagOfVisualWords model
        """
        self.bovw = bovw_model
        self.image_database = []  # List of (image_path, feature_vector) tuples
        self.feature_matrix = None  # Numpy array of all features for faster search


    def add_images_to_database(self, image_paths: List[Path], verbose=True):
        """
        Add images to the searchable database

        Args:
            image_paths: List of image paths to add
            verbose: Print progress
        """
        if verbose:
            print(f"Adding {len(image_paths)} images to database...")

        for i, img_path in enumerate(image_paths):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} images")

            features = self.bovw.get_bovw_features(img_path)
            self.image_database.append((img_path, features))

        # Create feature matrix for efficient batch operations
        self.feature_matrix = np.array([feat for _, feat in self.image_database])

        if verbose:
            print(f"✓ Database built with {len(self.image_database)} images")
            print(f"  Feature matrix shape: {self.feature_matrix.shape}")


    def find_similar_images(
        self,
        query_image: Path,
        top_k: int = 5,
        metric: BoVWMetric = BoVWMetric.COSINE,
        return_scores: bool = True
    ) -> List[Tuple[Path, float]]:
        """
        Find the most similar images to a query image

        Args:
            query_image: Path to query image
            top_k: Number of most similar images to return
            metric: Similarity metric ('cosine', 'euclidean', 'chi_square', 'histogram_intersection')
            return_scores: Whether to return similarity scores

        Returns:
            List of (image_path, similarity_score) tuples, sorted by similarity
        """

        assert self.image_database is not None, "Image database is empty. Call add_images_to_database first."
        assert self.feature_matrix is not None, "Feature matrix not built. Call add_images_to_database first."

        # Extract features from query image
        query_features = self.bovw.get_bovw_features(query_image)

        # Compute similarities to all images in database
        similarities = []

        if metric == BoVWMetric.COSINE:
            # Efficient batch computation using matrix multiplication
            scores = np.dot(self.feature_matrix, query_features)
            # Higher is better for cosine similarity
            for i, score in enumerate(scores):
                similarities.append((self.image_database[i][0], score))
            # Sort descending
            similarities.sort(key=lambda x: x[1], reverse=True)

        elif metric == BoVWMetric.EUCLIDEAN:
            # Compute Euclidean distance to all images
            distances = np.linalg.norm(self.feature_matrix - query_features, axis=1)
            for i, dist in enumerate(distances):
                similarities.append((self.image_database[i][0], dist))
            # Sort ascending (lower distance = more similar)
            similarities.sort(key=lambda x: x[1])

        elif metric == BoVWMetric.CHI_SQUARE:
            for img_path, features in self.image_database:
                dist = metric.compare(query_features, features)
                similarities.append((img_path, dist))
            # Sort ascending (lower distance = more similar)
            similarities.sort(key=lambda x: x[1])

        elif metric == BoVWMetric.HISTOGRAM_INTERSECTION:
            for img_path, features in self.image_database:
                score = metric.compare(query_features, features)
                similarities.append((img_path, score))
            # Sort descending (higher intersection = more similar)
            similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return similarities[:top_k]


    def is_similar_to_group(
        self,
        query_image: Path,
        threshold: float = 0.5,
        metric: BoVWMetric = BoVWMetric.COSINE,
        min_similar_count: int = 1
    ) -> Tuple[bool, float, List[Tuple[Path, float]]]:
        """
        Determine if a query image is similar to the group

        Args:
            query_image: Path to query image
            threshold: Similarity threshold (meaning depends on metric)
                      - cosine: typically 0.5-0.8 (higher = stricter)
                      - euclidean: typically 0.5-1.5 (lower = stricter)
                      - chi_square: typically 0.1-0.5 (lower = stricter)
            metric: Similarity metric to use
            min_similar_count: Minimum number of similar images needed to be "in group"

        Returns:
            Tuple of (is_similar, max_similarity, top_matches)
            - is_similar: Boolean indicating if image belongs to group
            - max_similarity: Best similarity score found
            - top_matches: Top 5 most similar images with scores
        """
        # Find top similar images
        top_matches = self.find_similar_images(query_image, top_k=5, metric=metric)

        if not top_matches:
            return False, 0.0, []

        # Count how many images meet the threshold
        if metric in ['cosine', 'histogram_intersection']:
            # Higher is better
            similar_count = sum(1 for _, score in top_matches if score >= threshold)
            max_similarity = top_matches[0][1]  # First is best for these metrics
        else:
            # Lower is better (euclidean, chi_square)
            similar_count = sum(1 for _, score in top_matches if score <= threshold)
            max_similarity = top_matches[0][1]

        is_similar = similar_count >= min_similar_count

        return is_similar, max_similarity, top_matches


    def batch_find_outliers(
        self,
        threshold: float = 0.3,
        metric: str = 'cosine'
    ) -> List[Tuple[Path, float]]:
        """
        Find images in the database that are dissimilar to all others (potential outliers)

        Args:
            threshold: Similarity threshold
            metric: Similarity metric to use

        Returns:
            List of (image_path, average_similarity) for potential outliers
        """
        outliers = []

        assert self.feature_matrix is not None, "Feature matrix not built. Call add_images_to_database first."

        for i, (img_path, img_features) in enumerate(self.image_database):
            # Compute similarity to all other images
            if metric == 'cosine':
                # Exclude self (index i)
                other_features = np.delete(self.feature_matrix, i, axis=0)
                similarities = np.dot(other_features, img_features)
                avg_similarity = np.mean(similarities)

                if avg_similarity < threshold:
                    outliers.append((img_path, avg_similarity))

            elif metric == 'euclidean':
                other_features = np.delete(self.feature_matrix, i, axis=0)
                distances = np.linalg.norm(other_features - img_features, axis=1)
                avg_distance = np.mean(distances)

                # For euclidean, high average distance means outlier
                if avg_distance > threshold:
                    outliers.append((img_path, avg_distance))

        # Sort by score
        if metric == 'cosine':
            outliers.sort(key=lambda x: x[1])  # Ascending (lowest similarity first)
        else:
            outliers.sort(key=lambda x: x[1], reverse=True)  # Descending (highest distance first)

        return outliers


    def save(self, filepath: str):
        """Save the similarity search database"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'image_database': self.image_database,
                'feature_matrix': self.feature_matrix
            }, f)


    def load(self, filepath: str):
        """Load the similarity search database"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.image_database = data['image_database']
            self.feature_matrix = data['feature_matrix']

