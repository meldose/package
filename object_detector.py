import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from test_pose_belt_conveyor import BoxPoseDetector


class ObjectDetector:
    def __init__(self, calibration_matrix_path=""):
        # self.T_cam_to_tcp = np.load(calibration_matrix_path)
        # self.T_cam_to_tcp = np.eye(4)
        # rot = R.from_quat(np.array([-0.022408655662149786, 0.99410398563951585, -0.10598323033833017, -0.0047615936509700016]))
        # rot_matrix = rot.as_matrix()  # 3x3

        # # Create 4x4 transformation matrix
        # self.T_cam_to_tcp[:3, :3] = rot_matrix
        # self.T_cam_to_tcp[:3, 3] = np.array([0.29957963417130595, 0.50912972601126771,0.60596833578701992]).T
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.profile = self.pipeline.start(cfg)
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)
        self.latest_color_image = None

        # Allow auto-exposure to settle
        for _ in range(5):
            self.pipeline.wait_for_frames()

    def get_detection(self, retry_count=5, delay_sec=0.5, visualize=True):
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

        # Create the PinholeCameraIntrinsic object
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(
            width=self.intrinsics.width,
            height=self.intrinsics.height,
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.ppx,
            cy=self.intrinsics.ppy
        )

        extcamcalib = np.eye(4)
        rot = R.from_quat(np.array([
            -0.022408657662149786, 
            0.99410398563951885, 
            -0.10598323033833017, 
            -0.0047618936509700016
            ]))
        rot_matrix = rot.as_matrix()
        extcamcalib[:3, :3] = rot_matrix
        extcamcalib[:3, 3] = np.array([
            0.29957963417130595, 
            0.50912972571126771,
            0.57596833578701992
        ]).T
        #
        bpd_ = BoxPoseDetector(intrinsics, extcamcalib)

        # TODO hardcoded values to be passed as config
        points = np.array([
            [0.0, -0.25, 0.18],
            [0.4, -0.25, 0.18],
            [0.0, -0.57, 0.18],
            [0.4, -0.57, 0.18],
            [0.0, -0.25, 0.50],
            [0.4, -0.25, 0.50],
            [0.0, -0.57, 0.50],
            [0.4, -0.57, 0.50],
        ])

        pointso3d = o3d.utility.Vector3dVector(points)

        # Create an OrientedBoundingBox from these points
        workspace = o3d.geometry.OrientedBoundingBox.create_from_points(pointso3d)
        workspace.color = [0.2, 0.8, 0.2]

        poses_mat_list = bpd_.detect_boxes(color_image, depth_image_raw, workspace)
        return poses_mat_list
    
        #     depth_colormap = cv2.convertScaleAbs(depth_image_raw, alpha=0.03)

        #     # Preprocess for contour detection
        #     gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #     edges = cv2.Canny(blurred, 50, 150)
        #     dilated = cv2.dilate(edges, None, iterations=2)

        #     contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     valid_contour = None

        #     for contour in contours:
        #         area = cv2.contourArea(contour)
        #         if area < 2000 or area > 20000:
        #             continue

        #         hull = cv2.convexHull(contour)
        #         hull_area = cv2.contourArea(hull)
        #         if hull_area == 0:
        #             continue
        #         solidity = area / float(hull_area)
        #         if solidity < 0.85:
        #             continue

        #         x, y, w, h = cv2.boundingRect(contour)
        #         aspect_ratio = w / float(h)
        #         if aspect_ratio < 0.8 or aspect_ratio > 1.2:
        #             continue

        #         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        #         if len(approx) < 4:
        #             continue

        #         rect = cv2.minAreaRect(contour)
        #         (center_x, center_y), _, angle = rect
        #         center_x, center_y = int(center_x), int(center_y)

        #         if not (0 <= center_x < depth_image_raw.shape[1] and 0 <= center_y < depth_image_raw.shape[0]):
        #             continue

        #         depth_value = depth_frame.get_distance(center_x, center_y)
        #         if depth_value <= 0 or np.isnan(depth_value):
        #             window = depth_image_raw[max(0, center_y - 2):center_y + 3,
        #                                     max(0, center_x - 2):center_x + 3]
        #             window = window[window > 0]
        #             if len(window) == 0:
        #                 self._show_no_object(color_image, show=visualize)
        #                 time.sleep(delay_sec)
        #                 continue
        #             depth_value = np.median(window) * 0.001  # Convert mm to meters

        #         if depth_value < 0.2 or depth_value > 1.5:  # Filter by expected object distance
        #             continue

        #         # Filter by depth variance inside the contour
        #         mask = np.zeros(depth_image_raw.shape, dtype=np.uint8)
        #         cv2.drawContours(mask, [contour], -1, 255, -1)
        #         depth_inside = depth_image_raw[mask == 255]
        #         depth_inside = depth_inside[depth_inside > 0]
        #         if len(depth_inside) < 10 or np.std(depth_inside) > 30:
        #             continue

        #         valid_contour = contour
        #         break  # First valid contour found

        #     if valid_contour is None:
        #         self._show_no_object(color_image, show=visualize)
        #         time.sleep(delay_sec)
        #         continue

        #     # Final pose estimation
        #     xyz_camera = rs.rs2_deproject_pixel_to_point(self.intrinsics, [center_x, center_y], depth_value)
        #     xyz_camera = np.array([*xyz_camera, 1])  # Homogeneous coordinates
        #     position_tcp = self.T_cam_to_tcp @ xyz_camera

        #     if visualize:
        #         self._show_detection(color_image, valid_contour, (center_x, center_y), angle)

        #     return {
        #         "pixel": (center_x, center_y),
        #         "position_camera": xyz_camera[:3],
        #         "position_tcp": position_tcp[:3],
        #         "orientation_deg": angle,
        #         "color_image": color_image,
        #         "depth_frame": depth_frame,
        #         "depth_image": depth_colormap
        #     }

        # print("[ObjectDetector] No object detected after retries.")
        # return None

    def _show_no_object(self, image, show=True):
        if show and image is not None:
            img = image.copy()
            cv2.putText(img, "No object found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)

    def _show_detection(self, image, contour, center, angle):
        img = image.copy()
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        cx, cy = center
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(img, f"Angle: {angle:.1f}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Object Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Detection Window",img)
        cv2.waitKey(1)

    def get_last_color_image(self):
        return self.latest_color_image
    
    def get_focal_length(self, average=False):
        return (self.intrinsics.fx + self.intrinsics.fy) / 2 if average else self.intrinsics.fx

    def release(self):
        self.pipeline.stop()
