
import cv2
import numpy as np
from cftracker.bacf import BACF
from cftracker.config import bacf_config #checked

class PyTracker:
    def __init__(self, video_path, tracker_type, dataset_config):
        self.video_path = video_path
        self.tracker_type = tracker_type
        self.video_capture = cv2.VideoCapture(video_path)
        self.init_gt = None

        if self.tracker_type == 'BACF':
            self.tracker = BACF(config=bacf_config.BACFConfig())                        
        else:
            raise NotImplementedError

    def select_roi(self, frame):
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        return roi

    def tracking(self, verbose=True, output_path=r"/home/accurate2/pyCFT/pyCFTrackers_bacf/output_videos/van_out.mp4"):
        poses = []
        success, init_frame = self.video_capture.read()

        if not success:
            print("Failed to read the video.")
            return

        if self.init_gt is None:
            # Use the select_roi function to interactively select the ROI
            self.init_gt = self.select_roi(init_frame)

        self.tracker.init(init_frame, self.init_gt)
        writer = None

        if verbose and output_path is not None:
            fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                     (init_frame.shape[1], init_frame.shape[0]))

        while True:
            success, current_frame = self.video_capture.read()
            if not success:
                break

            bbox = self.tracker.update(current_frame, vis=verbose)
            x, y, w, h = map(int, bbox)

            if verbose:
                # Draw rectangle around the tracked object
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                cv2.imshow('demo', current_frame)
                if writer is not None:
                    writer.write(current_frame)
                cv2.waitKey(1)

            poses.append(np.array([x, y, w, h]))

        self.video_capture.release()
        if writer is not None:
            writer.release()

        return np.array(poses)

# Example usage:
video_file = r'/home/accurate2/pyCFT/pyCFTrackers_bacf/videos/van.mp4'
tracker_type = 'BACF'
dataset_configuration = None  

tracker = PyTracker(video_file, tracker_type, dataset_configuration)
tracker.tracking()
