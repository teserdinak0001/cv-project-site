from extraction import Extractor
from segmentation import Cropper
from learning import DinoClusterer

class UnsupervisedLearner():
    def __init__(self, input_dir="input", frame_dir="extracted", temporal_window_size=30, 
                 crop_dir="crops", cluster_dir="clusters", preserve_data=True, max_videos=None, 
                 verbose=False) -> None:
        """Default constructor for the UnsupervisedLearner class.
        UnsupervisedLearner processes a directory coonsisting of .mp4 video files
        to find semantic patterns in the data. Input .mp4 files should be named
        in the format "YYYY-MM-DD_HH-MM-SS-front.mp4" where "YYYY-MM-DD" is the
        date of the video, "HH-MM-SS" is the time of the video, and "front" is
        the camera used from the Tesla dashcam system. The output of the UnsupervisedLearner 
        is a directory containing merged object crops belonging to subclasses with semantic meaning.
        Args:
            input_dir (str): The path to the input directory containing .mp4 files.
            frame_dir (str): The path to the output directory to save the frames from.mp4 files.
            crop_dir (str): The path to the output directory to save the object crops.
            temporal_window_size (int): The size of the window used to pool frames.
                + Frames from multiple files may overlap in time, the highest quality 
                frame in each window is retained.
            preserve_data (bool): When true entire dataset is overwritten, otherwise only new data is added.
                + Data is added from files that are present in the input directory at execution.
            max_videos (int): The maximum number of videos to process.
                + When not none a random sample of videos is selected.
            verbose (bool): When true, verbose output is printed to the console.
        Returns:
            None
        """
        self.input_dir = input_dir
        self.frame_dir = frame_dir
        self.crop_dir = crop_dir
        self.cluster_dir = cluster_dir
        self.temporal_window_size = temporal_window_size
        self.preserve_data = preserve_data
        self.max_videos = max_videos
        self.verbose = verbose

        self.extractor = Extractor(self.input_dir, self.frame_dir, preserve_data=self.preserve_data,
                                   temporal_window_size=self.temporal_window_size, max_videos=self.max_videos,
                                   verbose=self.verbose)
        
        self.cropper = Cropper(input_dir=self.frame_dir, output_dir=self.crop_dir, verbose=self.verbose)
        
        self.clusterer = DinoClusterer(self.crop_dir, self.cluster_dir, verbose=self.verbose)
        
    def extract_frames(self) -> None:
        """Extracts and adds or merges frames from .mp4 files in the input directory.
        The run method of the extractor class builds a .csv file containing metadata
        describing the dataset. Each row corresponds to a frame from a .mp4 file. Frames
        are scored based on an objective function in the extractor class, which is used
        to pick a frame from each pool. In addition, frames are saved to the output directory
        and are used later in the pipeline to generate object crops.
        Args:
            temporal_window_size (int): The size of the window used to pool frames.
                + Frames from multiple files may overlap in time, the highest quality 
                frame in each window is retained.
        Returns:
            None
        """
        self.extractor.run()

    def crop_objects(self):
        """Crops objects from frames and saves them to the output directory. The initial
        YOLO model is trained on two generic classes consisting of random sign like objects
        and traffic signals. A low confidence YOLO detection pass extracts potential objects.
        Args:
            None
        Returns:
            None
        """
        self.cropper.run()

    def cluster_objects(self):
        """Clusters object crops into semantic classes and saves them to the output directory.
        The pipeline uses self-supervised pretrained representations (DINO) combined with 
        unsupervised clustering to discover semantic subclasses. Clustering is performed with
        DBSCAN.
        Args:
            None
        Returns:
            None
        """
        self.clusterer.run()

    def run(self):
        """Runs the UnsupervisedLearner pipeline.
        Args:
            None
        Returns:
            None
        """
        self.extract_frames()
        self.crop_objects()
        self.cluster_objects()


if __name__ == "__main__":
    learner = UnsupervisedLearner(preserve_data=True, verbose=True)
    learner.run()
        
