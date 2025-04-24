#/usr/bin/env python3

from typing import List, Tuple
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
import cv2


def pcl_from_rgbd(
    color_img: np.ndarray,
    depth_mask: np.ndarray,
    intrins: o3d.camera.PinholeCameraIntrinsic,
    extrins: np.ndarray = np.eye(4),
    scale: float = 1000.0,
) -> o3d.geometry.PointCloud:
    depth_3d = o3d.geometry.Image(np.ascontiguousarray(depth_mask))
    rgb_img_copy = deepcopy(color_img)
    color_img_ = np.where(depth_mask[:, :, None], rgb_img_copy, 0)
    rgb_3d = o3d.geometry.Image(color_img_)
    rgbd_3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_3d,
        depth=depth_3d,
        depth_scale=scale,
        convert_rgb_to_intensity=False,
    )
    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd_3d, intrinsic=intrins, extrinsic=extrins
    )
    # pcl.estimate_normals()
    return pcl

def create_frame(
    translation: np.ndarray = np.zeros((3, 1), dtype=np.float32),
    orientation: np.ndarray = np.eye(3, dtype=np.float32),
    frame_size=0.1,
) -> o3d.geometry.TriangleMesh:
    frame_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=translation
    )
    frame_pose.rotate(orientation)
    return frame_pose


class BoxPoseDetector:
    def __init__(self, intrinsics:o3d.camera.PinholeCameraIntrinsic, extcam2root:np.ndarray):
        self.intrinsics_ = intrinsics
        self.extcam2root_ = extcam2root

    def detect_boxes(self, color:np.ndarray, depth: np.ndarray, workspace: o3d.geometry.OrientedBoundingBox) -> List[np.ndarray]:
        poses = []
        pcl = pcl_from_rgbd(
            color, depth, self.intrinsics_, self.extcam2root_, scale=1000.0
        )
        root_frame = create_frame(
            frame_size=0.05
        )
        pcl = pcl.crop(workspace)
        clusters = self.cluster_all_clouds(pcl)
        visu_geometries = [root_frame, pcl, workspace]
        for cluster in clusters:
            cluster_plane = self.plane_segmentation(cluster)[0]
            min_oobox = cluster_plane.get_oriented_bounding_box()
            min_oobox.color = np.random.rand(3)

            pose_mat = self.pose_from_obbox(min_oobox, z_up=False)
            poses.append(pose_mat)
            pose_frame = create_frame(
                translation=pose_mat[:3, 3],
                orientation=pose_mat[:3, :3],
                frame_size=0.03
            )
            visu_geometries.append(min_oobox)
            visu_geometries.append(pose_frame)
    
        # o3d.visualization.draw_geometries(visu_geometries)
        return poses

    # TODO hardcoded parameters to be passed as config
    @staticmethod
    def cluster_all_clouds(
        pcl_in: o3d.geometry.PointCloud,
        eps: float = 0.01,
        min_pts: int = 20,
    ) -> List[o3d.geometry.PointCloud]:
        pcl_in = pcl_in.voxel_down_sample(voxel_size=0.002)
        labels = np.array(pcl_in.cluster_dbscan(eps=eps, min_points=min_pts))
        
        clusters = []
        max_label = labels.max()

        for i in range(max_label + 1):
            indices = np.where(labels == i)[0]
            cluster = pcl_in.select_by_index(indices)
            clusters.append(cluster)
        
        return clusters

    # TODO hardcoded parameters to be passed as config
    @staticmethod
    def plane_segmentation(
        pcl_in: o3d.geometry.PointCloud,
        distance_threshold: float = 0.01,
        ransac_n: int = 5,
        num_iterations: int = 100,
        probability: float = 0.99
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        # First segmentation
        plane_model, inliers = pcl_in.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            # probability=probability,
        )
        inlier_cloud_1 = deepcopy(pcl_in.select_by_index(inliers))

        try:
            # Re-segment the remaining cloud
            box_side = pcl_in.select_by_index(inliers, invert=True)
            plane_model, inliers = box_side.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
                probability=probability,
            )
            inlier_cloud_2 = pcl_in.select_by_index(inliers)
        except Exception:
            return inlier_cloud_1, plane_model

        # Choose the larger inlier set
        if len(inlier_cloud_2.points) > len(inlier_cloud_1.points):
            inlier_cloud = inlier_cloud_2
        else:
            inlier_cloud = inlier_cloud_1

        return inlier_cloud, plane_model

    @staticmethod
    def pose_from_obbox(
        obbox: o3d.geometry.OrientedBoundingBox,
        z_up: bool = True,
    ) -> np.ndarray:
        """
        Get the pose of the bounding box center

        Parameters
        ----------
        obbox : o3d.geometry.OrientedBoundingBox
            Oriented Bounding Box
        z_up : bool, optional
            Z axis is towards up, by default True
        class_name : str, optional
            If defined we return the pose of the center of the box

        Returns
        -------
        np.ndarray
            Pose of the bounding box
        ///      ------- x
        ///     /|
        ///    / |
        ///   /  | z
        ///  y
        ///      0 ------------------- 1
        ///       /|                /|
        ///      / |               / |
        ///     /  |              /  |
        ///    /   |             /   |
        /// 2 ------------------- 7  |
        ///   |    |____________|____| 6
        ///   |   /3            |   /
        ///   |  /              |  /
        ///   | /               | /
        ///   |/                |/
        /// 5 ------------------- 4
        """
        vertices = np.asarray(obbox.get_box_points())

        # # Define edges of the cube
        edges = {
            0: [(0, 1), (0, 2), (0, 3)],
            1: [(0, 1), (1, 6), (1, 7)],
            2: [(0, 2), (2, 5), (2, 7)],
            3: [(0, 3), (3, 5), (3, 6)],
            4: [(4, 5), (4, 6), (4, 7)],
            5: [(2, 5), (3, 5), (4, 5)],
            6: [(1, 6), (3, 6), (4, 6)],
            7: [(1, 7), (2, 7), (4, 7)],
        }
        # Min X, Max Y, Max Z
        zero_corner_idx = np.argmin(
            -vertices[:, 0] - vertices[:, 2] - vertices[:, 1]
        )
        zero_edges = edges[zero_corner_idx]
        # Calculate edge lengths
        edge_lengths = np.linalg.norm(
            vertices[np.array(zero_edges)[:, 0]]
            - vertices[np.array(zero_edges)[:, 1]],
            axis=1,
        )

        # Find longest and shortest edges
        longest_edge_idx = np.argmax(edge_lengths)
        shortest_edge_idx = np.argmin(edge_lengths)

        # Define x-axis direction vector
        x_axis = (
            vertices[zero_edges[longest_edge_idx][1]]
            - vertices[zero_edges[longest_edge_idx][0]]
        )
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Define z-axis direction vector
        z_axis = (
            vertices[zero_edges[shortest_edge_idx][1]]
            - vertices[zero_edges[shortest_edge_idx][0]]
        )
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Calculate y-axis direction vector
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Build transformation matrix
        center = np.mean(vertices, axis=0)

        pose = np.eye(4)
        pose[:3, 3] = center
        pose[:3, 0] = x_axis

        if not z_up:
            if z_axis[2] > 0:
                pose[:3, 1] = -y_axis
                pose[:3, 2] = -z_axis
            else:
                pose[:3, 1] = y_axis
                pose[:3, 2] = z_axis
        else:
            if z_axis[2] < 0:
                pose[:3, 1] = -y_axis
                pose[:3, 2] = -z_axis
            else:
                pose[:3, 1] = y_axis
                pose[:3, 2] = z_axis

        return pose

if __name__=="__main__":
    # Define the camera intrinsics
    width = 1280
    height = 720
    fx = 913.126
    fy = 911.222
    cx = 641.536
    cy = 373.942

    # Create the PinholeCameraIntrinsic object
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)
    #
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

    color = cv2.imread("/home/hrg/Desktop/package/color_img.png")
    depth = cv2.imread("/home/hrg/Desktop/package/depth_image.png", cv2.IMREAD_ANYDEPTH)

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
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(pointso3d)
    obb.color = [0.2, 0.8, 0.2]

    poses_mat_list = bpd_.detect_boxes(color, depth, obb)
    print(poses_mat_list)