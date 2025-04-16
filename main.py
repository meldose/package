import numpy as np
import cv2
from neurapy.robot import Robot
from object_detector import ObjectDetector
from robot_controller import RobotController

# Fixed orientation for placement (RPY)
DESIRED_PLACEMENT_RPY = [0, np.pi, 0]  # example: place flat, flipped along Y

# Fixed location to place items
PLACEMENT_POSITION = [0.182, -0.450, 0.370]  # adjust based on your table layout

def main():
    # Initialize detector & robot
    r = Robot()
    r.set_mode("Automatic")
    r.move_joint("New_capture")

    calibration_matrix = r"/home/hrg/Desktop/package/cam_to_tcp_transform.npy"
    detector = ObjectDetector(calibration_matrix)
    robot_control = RobotController()

    try:
        while True:
            detection = detector.get_detection()
            if detection is None:
                print("[DEBUG] No object detected.")
                continue

            print("[DETECTED] Camera XYZ:", detection["position_camera"], "Angle deg:", detection["orientation_deg"])

            # Get current TCP pose and transform detection into base coordinates
            tcp_pose_current = robot_control.robot.get_tcp_pose()
            print(tcp_pose_current)
            T_tcp_to_base = robot_control.pose_to_matrix(tcp_pose_current, gripper_offset_z=-0.089)
            print(T_tcp_to_base)
            pos_cam_hom = np.array([*detection["position_camera"], 1])
            base_coords = T_tcp_to_base @ detector.T_cam_to_tcp @ pos_cam_hom
            print(base_coords)

            # Orientation from detection
            yaw_rad = np.deg2rad(detection["orientation_deg"])

            # Construct pickup pose with object orientation
            pickup_pose = [
                base_coords[0],
                base_coords[1],
                base_coords[2],
                0, np.pi, yaw_rad
            ]
            print("[PICKUP POSE]", pickup_pose)

            # Move to pick position
            robot_control.move_to_pose(pickup_pose, speed=0.2)
            robot_control.close_gripper()

            # Lift object
            lift_pose = pickup_pose.copy()
            lift_pose[2] += 0.1
            robot_control.move_to_pose(lift_pose, speed=0.2)

            # Define fixed placement pose
            place_pose = PLACEMENT_POSITION + DESIRED_PLACEMENT_RPY
            print("[PLACE POSE]", place_pose)

            # Move to place position and release
            robot_control.move_to_pose(place_pose, speed=0.2)
            robot_control.open_gripper()

            # Optional: move back up
            post_place_pose = place_pose.copy()
            post_place_pose[2] += 0.1
            robot_control.move_to_pose(post_place_pose, speed=0.2)

            # Visualization
            img = detection["color_image"]
            cv2.circle(img, detection["pixel"], 5, (0, 255, 0), -1)
            cv2.imshow("Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as ex:
        print("[ERROR]", ex)

    finally:
        detector.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
