from enum import IntEnum
import cv2
from cv2 import Feature2D, KeyPoint
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import pickle
from typing import Sequence

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
    def __init__(self, *, n_clusters=100, detector_type=FeatureDetectorType.SIFT):
        """
        Initialize BoVW model

        Args:
            n_clusters: Number of visual words in vocabulary
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.vocabulary = None
        self.feature_detector: Feature2D = detector_type.getector()

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
            Descriptors as a numpy array of shape (num_keypoints, descriptor_size)

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


    def build_vocabulary(self, image_paths, sample_size=None):
        """
        Build visual vocabulary by clustering descriptors

        Args:
            image_paths: List of paths to training images
            sample_size: Optional, subsample descriptors for faster training
        """
        print("Extracting descriptors from images...")
        all_descriptors = []

        for img_path in image_paths:
            keypoints, descriptors = self.extract_descriptors(img_path)  # Unpack the tuple
            if descriptors is not None and len(descriptors) > 0:  # Check descriptors, not keypoints
                all_descriptors.append(descriptors)

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


    def get_bovw_features(self, image_path):
        """
        Generate BoVW histogram for a single image

        Args:
            image_path: Path to image

        Returns:
            Histogram of visual words (feature vector)
        """
        if self.kmeans is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")

        keypoints, descriptors = self.extract_descriptors(image_path)  # Unpack the tuple

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        # Assign each descriptor to nearest cluster (visual word)
        labels = self.kmeans.predict(descriptors)

        # Create histogram
        histogram, _ = np.histogram(labels, bins=np.arange(self.n_clusters + 1))

        # Normalize
        histogram = histogram / (histogram.sum() + 1e-6)

        return histogram


    def save(self, filepath):
        """Save the model"""
        with open(filepath, 'wb') as f:
            pickle.dump({'kmeans': self.kmeans,
                        'vocabulary': self.vocabulary,
                        'n_clusters': self.n_clusters}, f)

    def load(self, filepath):
        """Load the model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.vocabulary = data['vocabulary']
            self.n_clusters = data['n_clusters']


if __name__ == "__main__":

    # 1. Collect your image paths
    home = Path.home()
    image_folder = home / Path("SDU/MasterThesis/Orthomosaics/pngs/")
    image_paths = image_folder.glob("*.png")  # or *.jpg, *.jpeg
    # Or use glob: image_paths = glob.glob("path/to/images/*.png")

    format_image_paths = [p for p in image_paths if p.is_file()]
    image_paths = format_image_paths
    print(f"Found {len(image_paths)} images", flush=True)

    # 2. Create BoVW model
    bovw = BagOfVisualWords(n_clusters=200)  # 200 visual words

    # 3. Build vocabulary from training images
    # Use all images or a subset for vocabulary building
    bovw.build_vocabulary(image_paths[:100], sample_size=50000)

    # 4. Save the vocabulary
    bovw.save("bovw_model.pkl")

    # 5. Extract features for all images
    features_list = []
    for img_path in image_paths:
        features = bovw.get_bovw_features(img_path)
        features_list.append(features)

    features_array = np.array(features_list)
    print(f"Feature matrix shape: {features_array.shape}")  # (n_images, n_clusters)

    # 6. Save features
    np.save("bovw_features.npy", features_array)

