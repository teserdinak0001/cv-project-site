from math import e
from multiprocessing import pool
import os
import csv
import cv2
import random
from matplotlib.pyplot import box
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dev_tools import Logging
    logger = Logging()
except:
    pass

class Extractor():

    def __init__(self, input_dir, output_dir, yolo_model="traffic_objects.pt", max_workers=8, 
                 max_videos=None, preserve_data=False, verbose=False, temporal_window_size=30) -> None:
        """Default constructor for the Extractor class.
        Extractor processes a directory coonsisting of .mp4 video files and creates a .csv file containing metadata
        describing the dataset. Each row corresponds to a frame from a .mp4 file. Frames are scored in objective()
        which is used to select the most desirable frame from each timepool. Frames are also saved to the output directory
        and are used later in the pipeline to generate object crops.
        Args:
            input_dir (str): The path to the input directory containing .mp4 files.
            output_dir (str): The path to the output directory to save the .mp4 files.
            yolo_model (str): The path to the YOLO model to use for object detection.
            max_workers (int): The maximum number of workers for threading.
            max_videos (int): The maximum number of videos to process. 
                + When not none a random sample of videos is selected.
            preserve_data (bool): When true entire dataset is overwritten, otherwise only new data is added.
            verbose (bool): When true, verbose output is printed to the console.
            temporal_window_size (int): The size of the window used to pool frames.
                + Frames from multiple files may overlap in time, the highest quality
        Returns:
            None
        """
        self.verbose = verbose

        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_videos = max_videos
        self.temporal_window_size = temporal_window_size
        self.top_percentage = 0.75

        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            self.device = "cpu"

        self.model = YOLO(yolo_model).to(self.device)

        self.max_workers = max_workers

        self.video_paths = self.get_video_paths(input_dir)
        self.images = []

        self.preserve_data = preserve_data

        if not self.preserve_data:
            self.export_meta([], "meta.csv")

    def get_video_paths(self, input_dir) -> list[str]:
        """Returns a list of video paths in the input directory.
            Args:
                input_dir (str): The path to the input directory containing .mp4 files.
            Returns:
                video_paths (list): A list of video paths in the input directory.
        """
        video_paths = []
        for f in os.listdir(input_dir):
            if f.lower().endswith(".mp4"):
                video_paths.append(os.path.join(input_dir, f))
        return video_paths
    
    def get_sharpness(self, image) -> float:
        """Computes image sharpness using the variance of the Laplacian. The Laplacian 
        operator highlights regions of rapid intensity change. Images with higher high-frequency 
        content (edges, fine detail) will produce a larger variance, indicating greater sharpness. 
        Blurry images tend to have lower variance.
        Args:
            image (np.ndarray): Input image in BGR format.
        Returns:
            float: Variance of the Laplacian. Higher values indicate sharper images (desirable).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance
    
    def get_box_sharpness(self, image, box) -> float:
        """Measures sharpness of a box using the get_sharpness() function, enabling
        sharpness to be measured for each object in an image.
        Args:
            image (np.ndarray): The input image to check for sharpness.
            box (tuple): The coordinates of the box.
        Returns:
            sharpness (float): Sharpness score.
        """
        x1, y1, x2, y2 = map(int, box) # Maps the four variables to integers from YOLO float values
        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return 0.0

        return self.get_sharpness(region)
    
    def is_duplicate(self,frame1, frame2, threshold=25) -> bool:
        """Determines if two frames are duplicates based on mean absolute difference.
        Args:
            frame1 (np.ndarray): The first frame to compare.
            frame2 (np.ndarray): The second frame to compare.
            threshold (float): The mean absolute difference threshold below 
            which the frames are considered duplicates.
        Returns:
            tuple: A tuple containing the mean absolute difference and a boolean 
            indicating if the frames are duplicates.
        """
        diff = np.mean(np.abs(frame1.astype("float") - frame2.astype("float")))
        return diff, diff < threshold
    
    def get_desirability_score(self, image, model) -> float:
        """Objective function for temporal image extraction across dataset.
        The function is maximized in each temporal window to form an optimal dataset.
        Args:
            image (np.ndarray): Image to be scored.
        Returns:
            desirability (float): Desirability score.
        """
        results = model(image, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return 0.0
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        h, w = image.shape[:2]

        desirability = 0.0

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box) # Maps the four variables to integers from YOLO float values

            # Ensure box is within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            box_area = (x2 - x1) * (y2 - y1)
            box_area_norm = np.clip(box_area / (h * w), 0, 1)

            # Avoid boxes that are invalid after transformation or too small
            if (x2 <= x1 or y2 <= y1) or box_area_norm < 0.001:
                continue

            sharpness = self.get_box_sharpness(image, (x1, y1, x2, y2))
            sharp_norm = np.clip(np.log1p(sharpness) / 4.5, 0, 1) ** 2

            # np.sqrt(box_area_norm) prevents large boxes from dominating desirability
            desirability += np.sqrt(box_area_norm) * sharp_norm

        return desirability

    def get_global_time(self, start_time, frame_idx, fps) -> datetime:
        """Converts frame index to global time based on start time and fps.
        Args:
            start_time (datetime): The start time of the video.
            frame_idx (int): The index of the frame.
            fps (float): The frames per second of the video.
        Returns:
            datetime: The global time corresponding to the frame index.
        """
        return start_time + timedelta(seconds=frame_idx / fps)
    
    def get_start_time(self, video_path) -> datetime:
        """Extracts start time from video name in format: YYYY-MM-DD_HH-MM-SS
        Args:
            video_path (str): The path to the video file.
        Returns:
            start_time (datetime): The start time of the video.
        """
        video_name = os.path.basename(video_path).replace('.mp4', '').replace('-front', '')
        start_time = datetime.strptime(video_name, "%Y-%m-%d_%H-%M-%S")
        return start_time
    
    def pool_frames(self, metadata, temporal_window_size=30, top_percentage=0.75) -> list[dict]:
        """Pools frames into windows of size temporal_window_size (seconds).
        Args:
            metadata (list): A list of dictionaries containing metadata for each frame.
            temporal_window_size (int): The size of the temporal window in seconds.
        Returns:
            pooled_frames (list): A list of dictionaries containing metadata for the most
            desirable frame in each window.
        """
        meta = metadata.copy()
        meta = [m for m in meta if m["desirability"] > 0]
        top_frames = []

        window_size = timedelta(seconds=temporal_window_size)
        start_time = min(m["global_time"] for m in meta)

        buckets = {}

        for m in meta:
            delta = m["global_time"] - start_time
            bucket_id = int(delta // window_size)
            buckets[bucket_id] = buckets.get(bucket_id, []) + [m]

        for i, b in enumerate(sorted(buckets)):
            frames = buckets[b]
            best = max(frames, key=lambda f: f["desirability"])
            
            if self.verbose:
                window_start = start_time + b * window_size
                window_end = window_start + window_size
                try:
                    logger.pprogress("Pooling frames", i+1, len(buckets))
                except:
                    print(f"Pooling {len(frames)} frames from {window_start} to {window_end}")
            
            top_frames.append(best)

        if len(top_frames) == 0:
            return []

        k = max(1, int(len(top_frames) * top_percentage))
        export_pool = sorted(top_frames, key=lambda f: f["desirability"], reverse=True)[:k]

        if self.verbose: print(f"Exporting {len(export_pool)} pools from {len(top_frames)} pools (top {top_percentage*100}%)")

        return export_pool
    
    def get_video_metadata(self, video_path) -> list[dict]:
        """Stores metadata for each .mp4 file in the form of a list of dictionaries.
        Args:
            video_path (str): The path to the .mp4 file.
        Returns:
            metadata (dict): A dictionary containing metadata for each frame.
                video_path (str): The path to the video
                frame_idx (int): The index of the frame
                global_time (datetime): The current time of the frame
                desirability (float): The desirability score of the frame
                fps (float): The frames per second of the video
        """ 
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        start = self.get_start_time(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            if frame_idx % 15 == 0:
                if self.verbose and frame_idx % 75 == 0: print(f"Processing frame {frame_idx}/{total_frames-1} from {video_path}")
                
                global_time = self.get_global_time(start, frame_idx, fps)
                desirability = self.get_desirability_score(frame, self.model)

                frames.append({
                    "video_path": video_path, 
                    "frame_idx": int(frame_idx), 
                    "global_time": global_time.isoformat(timespec="milliseconds"), 
                    "desirability": float(desirability),
                    "fps": float(fps)
                })

            frame_idx += 1

        cap.release()

        self.export_meta(frames, "meta.csv", "a")

        return frames
    
    def export_meta(self, meta_list, output_path, write_mode="w") -> None:
        """Exports metadata to a .csv file.
        Args:
            meta_list (list): A list of dictionaries containing metadata for each frame.
            output_path (str): The path to the output .csv file.
            write_mode (str): The mode to use when writing to the output file.
                + "w" overwrites the file, "a" appends to the file
        Returns:
            None
        """
        fields = ["video_path", "frame_idx", "global_time", "desirability", "fps"]
        
        file_exists = os.path.exists(output_path)
        file_empty = not file_exists or os.path.getsize(output_path) == 0

        with open(output_path, write_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            if write_mode == "w" or file_empty:
                writer.writeheader()
            
            writer.writerows(meta_list)

    def load_meta(self, input_path) -> list[dict]:
        """Loads metadata from a .csv file.
        Args:
            input_path (str): The path to the input .csv file.
        Returns:
            meta (list): A list of dictionaries containing metadata for each frame.
                video_path (str): The path to the video
                frame_idx (int): The index of the frame
                global_time (datetime): The current time of the frame
                desirability (float): The desirability score of the frame
                fps (float): The frames per second of the video
        """
        with open(input_path, "r") as f:
            reader = csv.DictReader(f)
            meta = list(reader)
            for m in meta:
                m["frame_idx"] = int(m["frame_idx"])
                m["desirability"] = float(m["desirability"])
                m["fps"] = float(m["fps"])
                m["global_time"] = datetime.fromisoformat(m["global_time"])
            return meta
        
    def export_frames_from_pools(self, pools, output_dir, recent_window=10) -> None:
        """Exports frames from a list of dictionaries containing metadata for each frame. Frames are
        deconflicted in time because pooling has been performed. A sliding window of recent frames
        is used to prevent duplicate frames from being exported. This is particularly important for
        periods where the vehicle is not moving.
        Args:
            pools (list): A list of dictionaries containing metadata for each frame after pooling.
            output_dir (str): The path to the output directory.
            recent_window (int): The number of recent frames to check for duplicates.
        Returns:
            None
        """
        meta = sorted(pools, key=lambda p: p["global_time"]) # This is required for duplicate detection to work as intended
        os.makedirs(output_dir, exist_ok=True)

        exported_frames = []

        if self.verbose: print(f"Processing {len(meta)} pools over a temporal window from {min(m['global_time'] for m in meta)} to {max(m['global_time'] for m in meta)}")

        for m in meta:
            if self.verbose:
                current_pool = meta.index(m)
                try:
                    logger.pprogress("Exporting frames", current_pool+1, len(meta))
                except:
                    print(f"Exporting frames {current_pool}/{len(meta)}")

            video_path = m["video_path"]
            video_name = os.path.basename(video_path).replace('.mp4', '').replace('-front', '')
            frame_idx = m["frame_idx"]
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, m["frame_idx"])
            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            h, w, _ = frame.shape
            frame = frame[:int(h * 0.925), :]

            dup = False
            for f in exported_frames[-recent_window:]:
                _, is_dup = self.is_duplicate(frame, f)
                if is_dup:
                    dup = True
                    break

            if dup:
                #if self.verbose: print(f"Duplicate frame detected at index {frame_idx} in {video_name}")
                continue

            out_path = os.path.join(output_dir,f"{video_name}_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)

            exported_frames.append(frame)
            if len(exported_frames) > recent_window:
                exported_frames.pop(0)

    def get_already_processed(self, meta_path='meta.csv', metadata=None) -> list[str]:
        """Retruns a list of processed files as a list of file paths.
        Args:
            meta_path (str): The path to the metadata file.
            metadata (list): A list of dictionaries containing metadata for each frame.
        Returns:
            processed_files (list): A list of unique file paths.
        """
        if metadata is None:
            meta = self.load_meta(meta_path)
        else:
            meta = metadata
        return list(set(m["video_path"] for m in meta))

    def run(self) -> None:
        """Runs the Extractor pipeline.
        Args:
            max_videos (int): The maximum number of videos to process.
            temporal_window_size (int): The size of the window used to pool frames.
                + Frames from multiple files may overlap in time, the highest quality 
                frame in each window is retained.
            output_dir (str): The path to the output directory.
        Returns:
            None
        """

        if os.path.isdir(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))

        video_paths = self.video_paths.copy() # Avoid mutating the original list

        # Limit video paths to files that have not already been processed
        processed_files = self.get_already_processed("meta.csv")
        orinal_len = len(video_paths)
        video_paths = [vp for vp in video_paths if vp not in processed_files]
        if self.verbose: print(f"Found {len(video_paths)} videos to process after removing {orinal_len - len(video_paths)} duplicates")

        master_meta_list = []

        max_workers = min(self.max_workers, os.cpu_count())

        video_subset = video_paths
        if self.max_videos is not None:
            max_videos = min(self.max_videos, len(video_paths))
            video_subset = random.sample(video_paths, max_videos)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.get_video_metadata, vp)
                for vp in video_subset
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    master_meta_list.extend(result)
                except Exception as e:
                    print("Error processing video:", e)

        meta = self.load_meta("meta.csv")
        if self.verbose: print(f"Loaded {len(meta)} frames from meta.csv")

        if self.verbose: print(f"Pooling frames into windows of {self.temporal_window_size} seconds")
        pooled_meta = self.pool_frames(meta, temporal_window_size=self.temporal_window_size, top_percentage=self.top_percentage)
        
        self.export_frames_from_pools(pooled_meta, self.output_dir)


if __name__ == "__main__":
    extractor = Extractor("input", "extracted", preserve_data=True, max_workers=8, verbose=True, yolo_model="traffic_objects.pt", 
                          max_videos=1, temporal_window_size=20)
    extractor.run()
    print(extractor.load_meta("meta.csv"))
    