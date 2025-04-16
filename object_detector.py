# import numpy as np # importing numpy 
# import cv2 # importing cv2 module
# import pyrealsense2 as rs # importing pyrealsense 
# import time # importing time module 

# class ObjectDetector:
#     def __init__(self, calibration_matrix_path):
#         try:
#             self.T_cam_to_tcp = np.load(calibration_matrix_path)
#         except FileNotFoundError:
#             raise ValueError(f"Calibration matrix file not found at: {calibration_matrix_path}")

#         self.pipeline = rs.pipeline()
#         cfg = rs.config()
#         cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#         cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#         self.profile = self.pipeline.start(cfg)
#         self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
#         self.align = rs.align(rs.stream.color)

#         self.latest_color_image = None

#     def get_detection(self, retry_count=5, delay_sec=0.5, show_when_none=True):
#         """Attempts to detect an object. Retries if none found."""
#         for attempt in range(retry_count):
#             frames = self.pipeline.wait_for_frames()
#             aligned_frames = self.align.process(frames)

#             color_frame = aligned_frames.get_color_frame()
#             depth_frame = aligned_frames.get_depth_frame()

#             if not color_frame or not depth_frame:
#                 continue

#             color_image = np.asanyarray(color_frame.get_data())
#             self.latest_color_image = color_image

#             gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#             _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             if not contours:
#                 self._show_no_object(color_image, show=show_when_none)
#                 time.sleep(delay_sec)
#                 continue

#             best_contour = max(contours, key=cv2.contourArea, default=None)
#             if best_contour is None or cv2.contourArea(best_contour) < 1000:
#                 self._show_no_object(color_image, show=show_when_none)
#                 time.sleep(delay_sec)
#                 continue

#             rect = cv2.minAreaRect(best_contour)
#             (center_x, center_y), _, angle = rect
#             center_x, center_y = int(center_x), int(center_y)

#             depth_value = depth_frame.get_distance(center_x, center_y)
#             if depth_value <= 0 or np.isnan(depth_value):
#                 self._show_no_object(color_image, show=show_when_none)
#                 time.sleep(delay_sec)
#                 continue

#             xyz_camera = rs.rs2_deproject_pixel_to_point(self.intrinsics, [center_x, center_y], depth_value)
#             xyz_camera = np.array([*xyz_camera, 1])

#             detection = {
#                 "pixel": (center_x, center_y),
#                 "position_camera": xyz_camera[:3],
#                 "orientation_deg": angle,
#                 "color_image": color_image,
#                 "depth_frame": depth_frame,
#             }
#             return detection

#         # After retries
#         print("[ObjectDetector] No object detected after retries.")
#         return None

#     def _show_no_object(self, image, show=True):
#         """Displays a message on screen when no object is found."""
#         if show and image is not None:
#             img = image.copy()
#             cv2.putText(img, "No object found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2)
#             cv2.imshow("Detection", img)
#             cv2.waitKey(1)

#     def get_last_color_image(self):
#         return self.latest_color_image

    
#     def __del__(self):
#         self.release()

#     def release(self):
#         self.pipeline.stop()
#         cv2.destroyAllWindows()

import numpy as np
import cv2
import pyrealsense2 as rs
import time

class ObjectDetector:
    def __init__(self, calibration_matrix_path, min_area=1000, min_depth_mm=200, max_depth_mm=1000):
        try:
            self.T_cam_to_tcp = np.load(calibration_matrix_path)
        except FileNotFoundError:
            raise ValueError(f"Calibration matrix file not found at: {calibration_matrix_path}")

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.profile = self.pipeline.start(cfg)
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)

        self.latest_color_image = None
        self.min_area = min_area
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm

    def get_detection(self, retry_count=5, delay_sec=0.5, show_when_none=True):
        for attempt in range(retry_count):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            self.latest_color_image = color_image
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Image preprocessing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Depth-based filtering
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_mask = (depth_image > self.min_depth_mm) & (depth_image < self.max_depth_mm)
            depth_mask = depth_mask.astype(np.uint8) * 255
            binary = cv2.bitwise_and(binary, binary, mask=depth_mask)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self._show_no_object(color_image, show=show_when_none)
                time.sleep(delay_sec)
                continue

            best_contour = None
            best_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity < 0.8:
                    continue
                if area > best_area:
                    best_contour = cnt
                    best_area = area

            if best_contour is None:
                self._show_no_object(color_image, show=show_when_none)
                time.sleep(delay_sec)
                continue

            rect = cv2.minAreaRect(best_contour)
            (center_x, center_y), _, angle = rect
            center_x, center_y = int(center_x), int(center_y)

            depth_value = depth_frame.get_distance(center_x, center_y)
            if depth_value <= 0 or not np.isfinite(depth_value):
                self._show_no_object(color_image, show=show_when_none)
                time.sleep(delay_sec)
                continue

            # Draw for debugging
            cv2.drawContours(color_image, [best_contour], -1, (0, 255, 0), 2)
            cv2.circle(color_image, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.imshow("Contours", color_image)
            cv2.imshow("Binary", binary)
            cv2.waitKey(1)

            xyz_camera = rs.rs2_deproject_pixel_to_point(self.intrinsics, [center_x, center_y], depth_value)
            xyz_camera = np.array([*xyz_camera, 1])

            detection = {
                "pixel": (center_x, center_y),
                "position_camera": xyz_camera[:3],
                "orientation_deg": angle,
                "color_image": color_image,
                "depth_frame": depth_frame,
            }
            return detection

        print("[ObjectDetector] No object detected after retries.")
        return None

    def _show_no_object(self, image, show=True):
        if show and image is not None:
            img = image.copy()
            cv2.putText(img, "No object found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)

    def get_last_color_image(self):
        return self.latest_color_image

    def __del__(self):
        self.release()

    def release(self):
        if hasattr(self, "pipeline"):
            self.pipeline.stop()
        cv2.destroyAllWindows()
