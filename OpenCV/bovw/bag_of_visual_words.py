from enum import IntEnum
import cv2
from cv2 import Feature2D, KeyPoint
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import pickle
from typing import Sequence, List

class FeatureDetectorType(IntEnum):
    SIFT = 0
    ORB = 1

    def getector(self, **kwargs) -> Feature2D:
        if self == FeatureDetectorType.SIFT:
            return cv2.SIFT.create(**kwargs)
        elif self == FeatureDetectorType.ORB:
            return cv2.ORB.create(nfeatures=500, **kwargs)
        else:
            raise ValueError("Unsupported Feature Detector Type")



class BagOfVisualWords:
    def __init__(
        self,
        *,
        n_clusters: int = 100,
        detector_type: FeatureDetectorType = FeatureDetectorType.SIFT,
        use_tfidf: bool = True
    ):
        """
        Initialize BoVW model

        Args:
            n_clusters: Number of visual words in vocabulary
            detector_type: Type of feature detector to use (SIFT or ORB)
            use_tfidf: Whether to use TF-IDF weighting (default: True)
        """
        self.n_clusters: int = n_clusters
        self.kmeans: MiniBatchKMeans | None = None
        self.vocabulary = None
        self.feature_detector: Feature2D = detector_type.getector()
        self.use_tfidf: bool = use_tfidf
        self.idf_weights = None  # Will store IDF weights for each visual word


    def extract_descriptors(
            self,
            image: cv2.typing.MatLike | Path,
            mask: cv2.typing.MatLike | None = None
    ) -> tuple[Sequence[KeyPoint], cv2.typing.MatLike]:
        """
        Extract feature descriptors from an image.

        Args:
            image:
                - cv2.typing.MatLike: Image to be processed.
                - pahtlib.Path: Path to the image file. If the image does not exist, a ValueError is raised.

            mask (cv2.typing.MatLike): Optional mask to specify regions of interest. The regions for which the features
            should be detected need to be non-zero in the mask. [detect](https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html#aa4e9a7082ec61ebc108806704fbd7887)

        Returns:
            Tuple of (keypoints, descriptors) where descriptors is a numpy array of shape (num_keypoints, descriptor_size)

        """
        if isinstance(image, Path):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Image at {image} could not be loaded.")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        else:
            img = image
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.feature_detector.detectAndCompute(img, mask)
        return keypoints, descriptors


    def build_vocabulary(self, image_paths: List[Path], sample_size=None):
        """
        Build visual vocabulary by clustering descriptors and compute IDF weights

        Args:
            image_paths: List of paths to training images
            sample_size: Optional, subsample descriptors for faster training
        """
        print("Extracting descriptors from images...")
        all_descriptors = []
        image_descriptors = []

        for img_path in image_paths:
            # Keypoints can be used to filter descriptors if needed by size or location
            keypoints, descriptors = self.extract_descriptors(img_path)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)
                image_descriptors.append(descriptors)

        if not all_descriptors:
            raise ValueError("No descriptors extracted from any images")

        # Stack all descriptors
        all_descriptors = np.vstack(all_descriptors)
        print(f"Total descriptors: {len(all_descriptors)}")

        # Optionally subsample for faster clustering
        if sample_size and len(all_descriptors) > sample_size:
            indices = np.random.choice(len(all_descriptors), sample_size, replace=False)
            all_descriptors = all_descriptors[indices]
            print(f"Subsampled to: {len(all_descriptors)}")

        # Cluster descriptors to create vocabulary
        print(f"Clustering into {self.n_clusters} visual words...")
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,
                                       random_state=42,
                                       batch_size=1000,
                                       verbose=1)
        self.kmeans.fit(all_descriptors)
        self.vocabulary = self.kmeans.cluster_centers_
        print("Vocabulary created!")

        # Calculate IDF weights if TF-IDF is enabled
        if self.use_tfidf:
            print("Calculating IDF weights...")
            self._compute_idf_weights(image_descriptors)
            print("IDF weights computed!")


    def _compute_idf_weights(self, image_descriptors):
        """
        Compute IDF (Inverse Document Frequency) weights for visual words

        IDF(word) = log(N / df(word))
        where N is total number of images and df(word) is number of images containing that word

        Args:
            image_descriptors: List of descriptor arrays, one per image
        """
        n_images = len(image_descriptors)
        word_document_count = np.zeros(self.n_clusters)  # Count how many images contain each word

        if self.kmeans is None:
            raise ValueError("KMeans model not trained. Cannot compute IDF weights.")

        for descriptors in image_descriptors:
            if descriptors is not None and len(descriptors) > 0:
                # Assign descriptors to visual words
                labels = self.kmeans.predict(descriptors)
                # Get unique words present in this image
                unique_words = np.unique(labels)
                # Increment document count for these words
                word_document_count[unique_words] += 1

        # Calculate IDF: log(N / df)
        # Add 1 to avoid division by zero for words that don't appear in any image
        self.idf_weights = np.log(n_images / (word_document_count + 1))
        # print(f"IDF weights log(N/n_i): {self.idf_weights}")


    def get_bovw_features(self, image_path, normalize=True):
        """
        Generate BoVW histogram for a single image with optional TF-IDF weighting

        Args:
            image_path: Path to image
            normalize: Whether to L2 normalize the final feature vector (default: True)

        Returns:
            Histogram of visual words (feature vector), optionally weighted by TF-IDF
        """
        if self.kmeans is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")

        keypoints, descriptors = self.extract_descriptors(image_path)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        # Assign each descriptor to nearest cluster (visual word)
        labels = self.kmeans.predict(descriptors)

        # Create histogram (term frequency)
        histogram, _ = np.histogram(labels, bins=np.arange(self.n_clusters + 1))

        # Convert to float for TF-IDF calculation
        histogram = histogram.astype(np.float64)

        # Apply TF-IDF weighting if enabled
        if self.use_tfidf:
            if self.idf_weights is None:
                raise ValueError("IDF weights not computed. This should have been done in build_vocabulary.")

            # TF-IDF: TF(word) * IDF(word)
            # TF is already the raw count (histogram)
            # Optionally normalize TF by total number of descriptors in image
            tf = histogram / (histogram.sum() + 1e-6)
            histogram = tf * self.idf_weights

        # L2 Normalize the final feature vector
        if normalize:
            norm = np.linalg.norm(histogram)
            if norm > 0:
                histogram = histogram / norm

        return histogram


    def save(self, filepath):
        """Save the model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'vocabulary': self.vocabulary,
                'n_clusters': self.n_clusters,
                'use_tfidf': self.use_tfidf,
                'idf_weights': self.idf_weights
            }, f)


    def load(self, filepath):
        """Load the model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.vocabulary = data['vocabulary']
            self.n_clusters = data['n_clusters']
            self.use_tfidf = data.get('use_tfidf', False)  # For backward compatibility
            self.idf_weights = data.get('idf_weights', None)



if __name__ == "__main__":
    home = Path.home()
    image_folder = home / Path("SDU/MasterThesis/Orthomosaics/pngs/")
    image_paths = list(image_folder.glob("*.png"))  # Convert to list to avoid consuming generator

    # Filter to only actual files
    image_paths = [p for p in image_paths if p.is_file()]
    print(f"Found {len(image_paths)} images", flush=True)

    # Create BoVW model with TF-IDF
    bovw = BagOfVisualWords(n_clusters=200, use_tfidf=True)  # 200 visual words with TF-IDF
    bovw.build_vocabulary(image_paths[:100], sample_size=50000)
    bovw.save("bovw_model.pkl")

    features_list = []
    for img_path in image_paths:
        features = bovw.get_bovw_features(img_path)
        features_list.append(features)

    features_array = np.array(features_list)
    print(f"Feature matrix shape: {features_array.shape}")  # (n_images, n_clusters)

    np.save("bovw_features.npy", features_array)

