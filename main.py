<<<<<<< HEAD

import numpy as np
import cv2
=======
import numpy as np # imported numpy
import cv2 # imported cv2 module
import os # imported os module
import time # imported time moudule
import logging
>>>>>>> 72a4b7401b562ef6545c74746f94910cc0d7f23b
from neurapy.robot import Robot
from object_detector import ObjectDetector
from robot_controller import RobotController

<<<<<<< HEAD
# Fixed orientation for placement (RPY)
DESIRED_PLACEMENT_RPY = [0, np.pi, 0]  # example: place flat, flipped along Y

# Fixed location to place items
PLACEMENT_POSITION = [0.3, 0.2, 0.15]  # adjust based on your table layout
=======
class NoObjectDetectedError(Exception): # class No object detected 
    pass

# Fixed location to place items
PLACEMENT_POSITION = [0.3, 0.2, 0.15]  # adjust based on your table layout


 ### MAIN FUNCTION ###
>>>>>>> 72a4b7401b562ef6545c74746f94910cc0d7f23b

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

<<<<<<< HEAD
            print("[DETECTED] Camera XYZ:", detection["position_camera"], "Angle deg:", detection["orientation_deg"])
=======
            tcp_pose_current = robot_control.robot.get_tcp_pose()
            print(tcp_pose_current)
            T_tcp_to_base = robot_control.pose_to_matrix(tcp_pose_current, gripper_offset_z=-0.091)
            print(T_tcp_to_base)
>>>>>>> 72a4b7401b562ef6545c74746f94910cc0d7f23b

            # Get current TCP pose and transform detection into base coordinates
            tcp_pose_current = robot_control.robot.get_tcp_pose()
            T_tcp_to_base = robot_control.pose_to_matrix(tcp_pose_current, gripper_offset_z=-0.087)
            pos_cam_hom = np.array([*detection["position_camera"], 1])
            base_coords = T_tcp_to_base @ detector.T_cam_to_tcp @ pos_cam_hom

            # Orientation from detection
            yaw_rad = np.deg2rad(detection["orientation_deg"])
<<<<<<< HEAD

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
=======
            target_pose = [base_coords[0], base_coords[1], base_coords[2], 0, np.pi, yaw_rad]
            print(target_pose)

            logging.info("Moving to target pose: %s", target_pose)
            robot_control.move_to_pose(target_pose, speed=0.2)

            # Visualize the detection and the calculated object dimensions
            img = detector.get_last_color_image()
            if img is not None:
                pixel = detection["pixel"]
                px, py = pixel

                # Dynamically calculate the bounding box size based on the object's pixel dimensions
                width_px = detection.get("bbox_width", 40)  # Default width if not provided
                height_px = detection.get("bbox_height", 40)  # Default height if not provided

                # Adjust the box size for better visibility
                scale_factor = 1  # You can adjust this if you want the box to be larger or smaller
                width_px = int(width_px * scale_factor)
                height_px = int(height_px * scale_factor)

                # Top-left and bottom-right coordinates for the bounding box
                top_left = (px - width_px // 2, py - height_px // 2)
                bottom_right = (px + width_px // 2, py + height_px // 2)

                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(img, "Detected", (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(img, f"Position: {np.round(detection['position_camera'], 3)} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Angle: {round(detection['orientation_deg'], 1)}°", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                width_px_real = bottom_right[0] - top_left[0]
                height_px_real = bottom_right[1] - top_left[1]

                cv2.putText(img, f"Size: {width_px_real}x{height_px_real} px", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Use depth from position_camera[2] for size estimation
                try:
                    depth = detection["position_camera"][2]
                    focal_length = detector.get_focal_length()  # Must exist in ObjectDetector
                    width_m = width_px_real * depth / focal_length
                    height_m = height_px_real * depth / focal_length
                    cv2.putText(img, f"Size: {round(width_m, 3)}x{round(height_m, 3)} m", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    logging.warning("Depth-based size estimation failed: %s", e)

                cv2.imshow("Detection Window", img)

            # Show depth image
            if "depth_image" in detection:
                cv2.imshow("Depth View", detection["depth_image"])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("User requested exit.")
>>>>>>> 72a4b7401b562ef6545c74746f94910cc0d7f23b
                break

    except Exception as ex:
        print("[ERROR]", ex)

    finally:
        detector.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
