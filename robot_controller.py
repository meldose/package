# robot_controller.py
import numpy as np
from neurapy.robot import Robot
from scipy.spatial.transform import Rotation as R
from object_detector import ObjectDetector
import time
class RobotController:
    def __init__(self, robot_ip=None):
        self.robot = Robot(robot_ip) if robot_ip else Robot()
        self.robot.set_mode("Automatic")

    def pose_to_matrix(self, tcp_pose, gripper_offset_z):
        """
        Converts a TCP pose [X,Y,Z,Roll,Pitch,Yaw] into a 4x4 transformation matrix
        and explicitly accounts for gripper offset along TCP's Z-direction.
        
        Args:
            tcp_pose (list): TCP pose as [X,Y,Z,roll,pitch,yaw] 
                            (translation in meters, rotation in radians).
            gripper_offset_z (float): Gripper offset in meters along TCP's Z-axis.
                                        Default is 0.100 m (100mm). Adjust as needed.
                                        
        Returns:
            np.array: 4x4 corrected homogeneous transformation matrix (base to gripper-tip).
        """
        translation = np.array(tcp_pose[:3])  # [X,Y,Z]
        rpy = np.array(tcp_pose[3:])          # [Roll,Pitch,Yaw]

        # Compute rotation matrix from Euler angles
        rotation_matrix = R.from_euler('xyz', rpy).as_matrix()

        # Original TCP transformation (base→TCP)
        T_tcp = np.eye(4)
        T_tcp[:3,:3] = rotation_matrix
        T_tcp[:3,3] = translation

        # Account explicitly for your gripper offset along the TCP's local Z-axis
        T_gripper_offset = np.eye(4)
        T_gripper_offset[:3, 3] = [0, 0, gripper_offset_z]  # Z offset of gripper w.r.t. TCP

        # Final transformation (base→gripper_tip)
        T_final = T_tcp @ T_gripper_offset

        return T_final

    def move_to_pose(self, pose_xyzrpy, speed=0.2):
        
        
        linear_property = {
            "speed": speed,
            "acceleration": 0.1,
            "jerk": 100,
            "rotation_speed": 1.57,
            "rotation_acceleration": 5.0,
            "rotation_jerk": 100,
            "blending": False,
            "target_pose": [pose_xyzrpy],
            "current_joint_angles": self.robot.get_current_joint_angles(),
            "weaving": False,
            "pattern": 1,
            "amplitude_left": 0.003,
            "amplitude_right": 0.003,
            "frequency": 1.5,
            "dwell_time_left": 0.0,
            "dwell_time_right": 0.0,
            "elevation": 0.0,
            "azimuth": 0.0
        }

        self.robot.move_linear_from_current_position(**linear_property)
        io_set = self.robot.set_tool_digital_outputs([1.0,0.0,0.0])
        time.sleep(1)
        self.robot.move_joint("New_capture")
        self.robot.move_joint("P50")
        self.robot.move_joint("P57")

    def move_robot_based_on_angle(self,yaw_rad):
        # calibration_matrix = r"/home/hrg/Documents/package_detection/cam_to_tcp_transform.npy"
        # detector=ObjectDetector(calibration_matrix)
        # robot_control=RobotController()
        # detection=detector.get_detection()
        # yaw_rad=np.deg2rad(detection["orientation_deg"])
        # tcp_pose_current=robot_control.robot.get_tcp_pose()
        # T_tcp_to_base=robot_control.pose_to_matrix(tcp_pose_current,gripper_offset_z=-0.087) # adjusting the offset for the gripper

        # pos_cam_hom=np.array([*detection["position_camera"],1])
        # base_coords=T_tcp_to_base @ detector.T_cam_to_tcp @ pos_cam_hom
        # target_pose=[base_coords[0],base_coords[1],base_coords[2],0,np.pi,yaw_rad]

        if yaw_rad == 0 or yaw_rad == 360:

            self.robot.move_joint([0.18229699489025253,
            -0.45074154634338454,
            0.36958859562606833,
            -0.001521776648733053,
            0.6085402412020895,
            0.7935212814262113,
            0.0006595128791383861]
            )
        elif yaw_rad == 90 or yaw_rad == 180:

            self.robot.move_joint([-1.1864124246453485,
                0.6649982733787552,
                0.8029329569691264,
                -0.00040783815381041584,
                1.6769492204432905,
                -1.555638154824762]
                )

        io_set = self.robot.set_tool_digital_outputs([0.0,1.0,0.0]) # open
        self.robot.move_joint("New_capture")
        self.robot.stop()