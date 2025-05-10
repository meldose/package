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

            color_frame = aligned_frames.get_color_frame() # setting the color frame
            depth_frame = aligned_frames.get_depth_frame() # setting the depth frame

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
