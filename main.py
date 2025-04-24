import numpy as np # import numpy module
import cv2 # import cv2 module
from neurapy.robot import Robot
from object_detector import ObjectDetector # importing object_detector
from robot_controller import RobotController # importing robot_controller
from scipy.spatial.transform import Rotation as R

def matrix_to_pose(matrix):
    # Extract translation
    x, y, z = matrix[:3, 3]

    # Extract rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert to Euler angles (roll, pitch, yaw) in radians
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    return [x, y, z, roll, pitch, yaw]

def main():
    r = Robot()
    r.set_mode("Automatic")
    r.move_joint("New_capture")


    calibration_matrix = r"/home/hrg/Documents/package_detection/cam_to_tcp_transform.npy"
    detector = ObjectDetector(calibration_matrix)
    robot_control = RobotController()

    try:
        box_poses = detector.get_detection(visualize=True)
        for pose_mat in box_poses:
            first_pose_euler = matrix_to_pose(pose_mat)
            print(f"my pose {first_pose_euler}")
            # robot_control.check_gripper_status()
            robot_control.move_to_pose(first_pose_euler, speed=0.06)

    except Exception as ex:
        print("[ERROR]", ex)
    finally:
        detector.release()

if __name__ == "__main__":
    main()


