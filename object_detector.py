import numpy as np
import cv2
import pyrealsense2 as rs
import time


class ObjectDetector:
    def __init__(self, calibration_matrix_path, visualize=True):
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

        self.visualize = visualize
        self.latest_color_image = None

    def get_detection(self, retry_count=5, delay_sec=0.5, show_when_none=True):
        """Attempts to detect an object. Retries if none found."""
        for attempt in range(retry_count):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            self.latest_color_image = color_image
            depth_image_raw = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.convertScaleAbs(depth_image_raw, alpha=0.03)

            # Preprocess for contour detection
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            dilated = cv2.dilate(edges, None, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contour = None

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue

                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) < 4:
                    continue

                valid_contour = contour
                break  # Take the first valid one

            if valid_contour is None:
                self._show_no_object(color_image, show=show_when_none)
                time.sleep(delay_sec)
                continue

            rect = cv2.minAreaRect(valid_contour)
            (center_x, center_y), _, angle = rect
            center_x, center_y = int(center_x), int(center_y)

            depth_value = depth_frame.get_distance(center_x, center_y)
            if depth_value <= 0 or np.isnan(depth_value):
                window = depth_image_raw[center_y-2:center_y+3, center_x-2:center_x+3]
                window = window[window > 0]
                if len(window) == 0:
                    self._show_no_object(color_image, show=show_when_none)
                    time.sleep(delay_sec)
                    continue
                depth_value = np.median(window) * 0.001  # Convert from mm to meters if needed

            # Convert the pixel (center_x, center_y) and depth value to 3D coordinates in camera space
            xyz_camera = rs.rs2_deproject_pixel_to_point(self.intrinsics, [center_x, center_y], depth_value)
            xyz_camera = np.array([*xyz_camera, 1])  # Homogeneous coordinates

            # Transform the camera coordinates to the robot's TCP frame
            position_tcp = self.T_cam_to_tcp @ xyz_camera

            if self.visualize:
                debug_image = color_image.copy()
                cv2.drawContours(debug_image, [valid_contour], -1, (0, 255, 0), 2)
                cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(debug_image, f"Depth: {depth_value:.3f}m", (center_x + 10, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imshow("Detection", debug_image)
                cv2.imshow("Depth Image", depth_colormap)
                cv2.waitKey(1)

            return {
                "pixel": (center_x, center_y),
                "position_camera": xyz_camera[:3],  # 3D coordinates in camera space
                "position_tcp": position_tcp[:3],   # Transformed to robot's TCP coordinates
                "orientation_deg": angle,  # Orientation of the detected object
                "color_image": color_image,
                "depth_frame": depth_frame,
                "depth_image": depth_colormap
            }

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

    def get_focal_length(self):
        return self.intrinsics.fx  # Can also return (fx + fy) / 2 if needed

    def __del__(self):
        self.release()

    def release(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            print(f"[ObjectDetector] Error during release: {e}")
        cv2.destroyAllWindows()
