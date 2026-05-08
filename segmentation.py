import os
import cv2
import random
import matplotlib.pyplot as plt

from ultralytics import YOLO

try:
    from dev_tools import Logging
    logger = Logging()
except:
    pass

class Cropper():

    def __init__(self, yolo_model="traffic_objects.pt", input_dir="extracted", output_dir="crops",
                 verbose=False) -> None:
        """Default constructor for the Cropper class.
        Args:
            yolo_model (str): The path to the YOLO model to use for object detection.
            input_dir (str): The path to the input directory containing full frames.
            output_dir (str): The path to the output directory to save the object crops.
            verbose (bool): When true, verbose output is printed to the console.
        Returns:
            None
        """
        self.model = YOLO(yolo_model, verbose=False)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.verbose = verbose

        os.makedirs(self.output_dir, exist_ok=True)

    def is_blurry(self, image, threshold=5000) -> bool:
        """Checks if an image is blurry based on its Laplacian variance.
        Args:
            image (np.ndarray): Input image in BGR format.
            threshold (float): The threshold for blurry image detection.
        Returns:
            bool: True if the image is blurry, False otherwise.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        h, w = gray.shape
        area = h * w

        threshold = max(250, threshold * (area / (640 * 480)))  # Scale threshold based on image area

        quality = True
        if gray.shape[0] < 60 or gray.shape[1] < 60 or variance < threshold:
            quality = False

        return variance, threshold, quality
    
    def laplacian_variance(self, image) -> float:
        """Computes image sharpness using the variance of the Laplacian.
        Args:
            image (np.ndarray): Input image in BGR format.
        Returns:
            float: Variance of the Laplacian. Higher values indicate sharper images (desirable).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def process_input(self, input_dir) -> None:
        """Processes the input directory and crops objects from full frames from video.
        Args:
            input_dir (str): The path to the input directory containing full frames.
        Returns:
            None
        """
        frames = os.listdir(input_dir)
        
        for frame in frames:
            if self.verbose: 
                try:
                    logger.pprogress("Processing frames", frames.index(frame)+1, len(frames))
                except:
                    print(f"Cropping objects: {frames.index(frame)}/{len(frames)}")

            self.crop(frame, self.model)

    def crop(self, img_filename, model, output_dir="crops") -> None:
        """Crops objects from a full frame and saves them to the output directory.
        Args:
            img_filename (str): The filename of the full frame image.
            model (YOLO): The YOLO model to use for object detection.
            output_dir (str): The path to the output directory to save the object crops.
        Returns:
            None
        """
        img_path = os.path.join(self.input_dir, img_filename)
        img = cv2.imread(img_path)

        img_h, img_w = img.shape[:2]
        results = model(img_path, conf=0.50, verbose=False)

        for i, result in enumerate(results):
            boxes = result.boxes.xyxy
            class_ids = result.boxes.cls
            confs = result.boxes.conf

            for j, (box, cls_id, conf) in enumerate(zip(boxes, class_ids, confs)):
                x1, y1, x2, y2 = map(int, box)

                class_name = model.names[int(cls_id)]

                crop = img[y1:y2, x1:x2]
                _, _, is_blurry_result = self.is_blurry(crop)

                if not is_blurry_result:
                    crop_filename = f"{os.path.splitext(img_filename)[0]}_{class_name}_{i}_{j}.jpg"
                    cv2.imwrite(os.path.join(output_dir, crop_filename), crop)

    def visualize(self, crop_dir="crops") -> None:
        """Shows a random visualization of top (k=50) crops based on Laplacian variance.
        Args:
            crop_dir (str): The path to the directory containing the object crops.
        Returns:
            None
        """
        crop_files = os.listdir(crop_dir)

        scored = []
        for f in crop_files:
            path = os.path.join(crop_dir, f)
            img = cv2.imread(path)
            if img is None:
                continue

            var = self.laplacian_variance(img)
            scored.append((f, var))

        scored.sort(key=lambda x: x[1], reverse=True)

        top_k = scored[:50]
        selected = random.sample(top_k, min(8, len(top_k)))

        fig, axes = plt.subplots(2, 4, figsize=(8, 4))  # smaller figure

        for idx, (crop_file, var) in enumerate(selected):
            crop_path = os.path.join(crop_dir, crop_file)
            crop_img = cv2.imread(crop_path)

            ax = axes[idx // 4, idx % 4]
            ax.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), interpolation='nearest')
            ax.set_title(f"{var:.1f}", fontsize=8)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def run(self) -> None:
        """Executes the object cropping and visualization process.
        Args:
            None
        Returns:
            None
        """
        if os.path.isdir(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))

        self.process_input(self.input_dir)
        #self.visualize()

if __name__ == "__main__":
    cropper = Cropper(verbose=True)
    cropper.run()