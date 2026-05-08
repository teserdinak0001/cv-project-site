import os
import shutil
import cv2
import csv
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_distances

try:
    from dev_tools import Logging
    logger = Logging()
except:
    pass

class DinoClusterer:

    def __init__(self, input_dir, output_dir="clusters", max_images=None, eps=0.10, min_samples=5, 
                 metric='cosine', merge_eps=0.15, verbose=False) -> None:
        """Default constructor for the DinoClusterer class.
        Args:
            input_dir (str): The path to the input directory containing object crops.
            output_dir (str): The path to the output directory to save the clusters.
            max_images (int): The maximum number of images to process.
            eps (float): The epsilon value for DBSCAN clustering.
            min_samples (int): The minimum number of samples required to form a cluster.
            metric (str): The metric to use for DBSCAN clustering.
                + Options: ['cosine', 'euclidean']
            merge_eps (float): The epsilon value for merging similar clusters.
        Returns:
            None
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_images = max_images
        self.verbose = verbose

        self.paths = []
        self.features = None
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.merge_eps = merge_eps

        self.transform = None

    def load_model(self) -> None:
        """Loads the DINO model.
        Args:
            None
        Returns:
            None
        """
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model = self.model.to(self.device).eval()

    def build_transform(self) -> T.Compose:
        """Builds the image transformation.
        Args:
            None
        Returns:
            transform (torchvision.transforms.Compose): The image transformation.
        """
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

        return transform

    def embed(self, path) -> np.ndarray:
        """Embeds an image using the DINO model. Converts an N x M x 3 image to
        a N x 3 vector. Dino embeddings are semantic and not purely visual.
        A gaussian blur is applied to the image to reduce background noise.
        Args:
            path (str): The path to the image file.
        Returns:
            feat (np.ndarray): The embedded feature vector.
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        img = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]

        img = cv2.GaussianBlur(img, (5, 5), 0)

        img = Image.fromarray(img)

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.forward_features(x)["x_norm_clstoken"]

        feat = feat.squeeze().cpu().numpy()

        return feat / (np.linalg.norm(feat) + 1e-8)

    def load_data(self) -> None:
        """Loads the image data from the input directory.
        Args:
            None
        Returns:
            None
        """
        print("Loading images...")

        self.paths = [
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]

        if self.max_images:
            self.paths = self.paths[:self.max_images]

        print(f"Found {len(self.paths)} images")

    def build_features(self) -> None:
        """Forms features from the object crops using the DINO model.
        Args:
            None
        Returns:
            None
        """
        print("Extracting embeddings...")

        features = []
        valid_paths = []

        for p in self.paths:
            try:
                feat = self.embed(p)
                features.append(feat)
                valid_paths.append(p)
            except:
                continue

        self.features = np.array(features)
        self.paths = valid_paths

        print("Feature shape:", self.features.shape)
   
    def cluster(self, features, eps=0.10, min_samples=5, metric='cosine') -> tuple[float, np.ndarray]:
        """Performs clustering on a set of features using DBSCAN.
        Args:
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            metric (str): The distance metric to use for clustering.
        Returns:
            labels (np.ndarray): An array of cluster labels for each sample.
        """
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

        labels = clusterer.fit_predict(features)
        objective = self.objective(features, labels)
        print(f"Objective Score: {self.objective(features, labels):.4f}")

        return objective, labels
    
    def merge_clusters(self, labels, merge_eps=0.15) -> np.ndarray:
        """Second-stage clustering on cluster centroids to merge similar clusters.
        Args:
            labels (np.ndarray): An array of cluster labels.
        Returns:
            labels (np.ndarray): An array of merged cluster labels.
        """
        labels = np.asarray(labels)

        # Step 1: compute centroids
        centroids = []
        cluster_ids = []

        for cid in sorted(set(labels)):
            if cid == -1:
                continue

            idx = np.where(labels == cid)[0]
            if len(idx) < 2:
                continue

            X = self.features[idx]
            centroid = X.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

            centroids.append(centroid)
            cluster_ids.append(cid)

        centroids = np.vstack(centroids)

        # Step 2: cluster centroids
        meta_clusterer = DBSCAN(eps=merge_eps, min_samples=1, metric='cosine')
        meta_labels = meta_clusterer.fit_predict(centroids)

        # Step 3: map original labels → merged labels
        new_labels = -np.ones_like(labels)

        for cid, meta in zip(cluster_ids, meta_labels):
            new_labels[labels == cid] = meta

        return new_labels

    def save_all_clusters(self, labels, output_dir="clusters") -> None:
        """Saves a grid of images for each cluster.
        Args:
            labels (np.ndarray): An array of cluster labels.
        Returns:
            None
        """
        labels = np.asarray(labels)
        os.makedirs(output_dir, exist_ok=True)

        for cid in sorted(set(labels)):
            if cid == -1:
                continue

            idxs = np.where(labels == cid)[0][:16]

            cols = 4
            rows = int(np.ceil(len(idxs) / cols))

            plt.figure(figsize=(8, 2 * rows))

            for i, idx in enumerate(idxs):
                img = cv2.imread(self.paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.axis("off")

            plt.suptitle(f"Cluster {cid}")
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f"cluster_{cid}.png"))
            plt.close()

    def save_cluster_crops(self, labels, output_dir="clusters") -> None:
        """Saves clusters to separate directories.
        Args:
            labels (np.ndarray): An array of cluster labels.
        Returns:
            None
        """
        labels = np.asarray(labels)
        os.makedirs(output_dir, exist_ok=True)

        for cid in sorted(set(labels)):
            if cid == -1:
                continue

            cluster_dir = os.path.join(output_dir, f"cluster_{cid}")
            os.makedirs(cluster_dir, exist_ok=True)

            idxs = np.where(labels == cid)[0]

            for i, idx in enumerate(idxs):
                src = self.paths[idx]

                # Keep original filename (safer for debugging)
                filename = os.path.basename(src)

                # Optional: prefix with index to avoid collisions
                dst = os.path.join(cluster_dir, f"{i:04d}_{filename}")

                try:
                    shutil.copy(src, dst)
                except:
                    continue

        print(f"Saved clusters to: {output_dir}")

    def objective(self, features, labels, min_samples=5) -> float:
        """Objective function to maximize for hyperparameter tuning.
        Args:
            None
        Returns:
            None
        """
        labels = np.asarray(labels)
        features = np.asarray(features)

        valid_labels = []
        centroids = []
        within_scores = []

        for label in sorted(set(labels)):
            if label == -1:
                continue

            idx = np.where(labels == label)[0]

            if len(idx) < min_samples:
                continue

            X = features[idx]

            centroid = X.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

            dists = cosine_distances(X, centroid.reshape(1, -1)).ravel()

            valid_labels.append(label)
            centroids.append(centroid)
            within_scores.append(dists.mean())

        n_clusters = len(valid_labels)

        if n_clusters < 2:
            return -np.inf

        avg_within = float(np.mean(within_scores))

        centroids = np.vstack(centroids)
        between = cosine_distances(centroids)
        upper = between[np.triu_indices_from(between, k=1)]
        avg_between = float(np.mean(upper))

        cluster_sizes = [len(np.where(labels == cid)[0]) for cid in valid_labels]
        dominance = max(cluster_sizes) / sum(cluster_sizes)

        score = (
            np.log1p(n_clusters)
            * (avg_between / (avg_within + 1e-12))
            * (1 - dominance)
        )
            
        return score
    
    def tune(self) -> dict:
        """Tunes hyperparameters for clustering.
        Args:
            None
        Returns:
            best_config (dict): A dictionary containing the best hyperparameters.
        """
        if not self.transform:
            self.load_model()
            self.transform = self.build_transform()
            self.load_data()
            self.build_features()

        eps = np.linspace(0.01, 0.35, 20)
        min_samples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        metric = ['cosine', 'euclidean']

        best_config = {}

        for e in eps:
            for m in metric:
                for ms in min_samples:
                    score, labels = self.cluster(self.features, eps=e, min_samples=ms, metric=m)
                    if score > best_config.get('score', 0):
                        best_config['score'] = score
                        best_config['eps'] = e
                        best_config['min_samples'] = ms
                        best_config['metric'] = m
                        print(f"New best config: {best_config}")

        print(f"Best config: {best_config}")
        self.eps = best_config['eps']
        self.min_samples = best_config['min_samples']
        self.metric = best_config['metric']

        return best_config
    
    def evaluate_clusters(self, labels, min_cluster_size=2) -> list:
        """Evaluates clustering on a set of object crops.
        Args:
            labels (np.ndarray): An array of cluster labels for each sample.
        Returns:
            results (list): A list of evaluation results.
        """
        labels = np.asarray(labels)
        features = np.asarray(self.features)

        results = []

        valid_clusters = []
        centroids = []
        within_scores = {}

        # --- Compute centroids + within distances ---
        for cid in sorted(set(labels)):
            if cid == -1:
                continue

            idx = np.where(labels == cid)[0]
            if len(idx) < min_cluster_size:
                continue

            X = features[idx]

            centroid = X.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

            dists = cosine_distances(X, centroid.reshape(1, -1)).ravel()
            avg_within = float(np.mean(dists)) / 2.0  # normalized

            valid_clusters.append(cid)
            centroids.append(centroid)
            within_scores[cid] = avg_within

        if len(valid_clusters) < 2:
            print("Not enough clusters for evaluation")
            return

        # --- Compute between distances ---
        centroids = np.vstack(centroids)
        between = cosine_distances(centroids)
        upper = between[np.triu_indices_from(between, k=1)]
        avg_between = float(np.mean(upper)) / 2.0  # normalized

        # --- Build table ---
        total_samples = len(labels)

        for cid in valid_clusters:
            size = np.sum(labels == cid)
            avg_within = within_scores[cid]

            ratio = avg_between / (avg_within + 1e-12)

            results.append({
                "cluster": cid,
                "size": int(size),
                "percent": size / total_samples,
                "avg_within": avg_within,
                "avg_between": avg_between,
                "ratio": ratio
            })

        # Sort by size (most important clusters first)
        results = sorted(results, key=lambda x: x["size"], reverse=True)

        return results

    def print_evaluation(self, results) -> None:
        """Prints evaluation results.
        Args:
            results (list): A list of evaluation results.
        Returns:
            None
        """
        print(f"{'Cluster':<8} {'Size':<6} {'%':<6} {'Within':<10} {'Between':<10} {'Ratio':<10}")
        print("-" * 60)

        for r in results:
            print(f"{r['cluster']:<8} {r['size']:<6} {r['percent']:.2f}   {r['avg_within']:.4f}   {r['avg_between']:.4f}   {r['ratio']:.2f}")

    def save_evaluation(self, results, path="cluster_eval.csv") -> None:
        """Saves evaluation results to a .csv file.
        Args:
            results (list): A list of evaluation results.
            path (str): The path to save the .csv file to.
        Returns:
            None
        """
        if not results:
            print("No results to save.")
            return

        # Get headers from keys of first result dict
        fieldnames = list(results[0].keys())

        with open(path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"Saved evaluation to {path}")

    def run(self) -> None:
        """Executes clustering on a set of object crops.
        Args:
            None
        Returns:
            None
        """
        for path in [self.output_dir, f"{self.output_dir}_merged"]:
            if os.path.exists(path):
                shutil.rmtree(path)

        if not self.transform:
            self.load_model()
            self.transform = self.build_transform()
            self.load_data()
            self.build_features()

        score , labels = self.cluster(self.features, eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.save_all_clusters(labels)
        self.save_cluster_crops(labels)
        results = self.evaluate_clusters(labels)
        self.print_evaluation(results)
        self.save_evaluation(results)

        labels = self.merge_clusters(labels, merge_eps=self.merge_eps)
        self.save_all_clusters(labels, output_dir="clusters_merged")
        self.save_cluster_crops(labels, output_dir="clusters_merged")
        results = self.evaluate_clusters(labels)
        self.print_evaluation(results)
        self.save_evaluation(results, path="cluster_eval_merged.csv")

        return score, labels

if __name__ == "__main__":
    clusterer = DinoClusterer(input_dir="crops", merge_eps=0.15)
    config = clusterer.tune()
    clusterer.run()