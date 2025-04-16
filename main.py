import numpy as np # import numpy as np
import cv2 # import cv2 module
import os # import os module 
import time # import time
import logging # import logging module 
from neurapy.robot import Robot
from object_detector import ObjectDetector # import ObjectDetector module
from robot_controller import RobotController # import RobotController 

class NoObjectDetectedError(Exception):
    """Raised when object detection fails after multiple retries."""
    pass

# Setup logging configuration
logging.basicConfig(
    filename='detection.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    r = Robot()
    r.set_mode("Automatic")
    r.move_joint("New_capture")
    calibration_matrix = (r"/home/hrg/Desktop/package/cam_to_tcp_transform.npy")
    if not os.path.exists(calibration_matrix):
        logging.error("Calibration matrix not found at: %s", calibration_matrix)
        raise FileNotFoundError(f"[ERROR] Calibration matrix not found at: {calibration_matrix}")

    detector = ObjectDetector(calibration_matrix)
    robot_control = RobotController()

    max_attempts = 30
    no_detection_count = 0

    try:
        while True:
            detection = detector.get_detection()

            if detection is None:
                no_detection_count += 1
                logging.warning("No object detected. Attempt %d/%d", no_detection_count, max_attempts)

                img = detector.get_last_color_image()
                if img is not None:
                    cv2.putText(img, "Scanning...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Detection Window", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logging.info("User quit during scanning phase.")
                        break
                    elif key == ord('p'):
                        logging.info("Paused by user.")
                        cv2.waitKey(0)

                if no_detection_count >= max_attempts:
                    logging.critical("Exceeded max detection attempts without success.")
                    raise NoObjectDetectedError("Failed to detect object after multiple retries.")
                continue
            else:
                logging.info("Object detected at camera XYZ: %s with angle %s", 
                             detection["position_camera"], detection["orientation_deg"])
                no_detection_count = 0

            tcp_pose_current = robot_control.robot.get_tcp_pose()
            T_tcp_to_base = robot_control.pose_to_matrix(tcp_pose_current, gripper_offset_z=-0.091)

            pos_cam_hom = np.array([*detection["position_camera"], 1])
            base_coords = T_tcp_to_base @ detector.T_cam_to_tcp @ pos_cam_hom

            yaw_rad = np.deg2rad(detection["orientation_deg"])
            target_pose = [base_coords[0], base_coords[1], base_coords[2], 0, np.pi, yaw_rad]

            logging.info("Moving to target pose: %s", target_pose)
            robot_control.move_to_pose(target_pose, speed=0.2)

            img = detector.get_last_color_image()
            if img is not None:
                pixel = detection["pixel"]
                px, py = pixel
                box_size = 40
                top_left = (px - box_size // 2, py - box_size // 2)
                bottom_right = (px + box_size // 2, py + box_size // 2)

                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(img, "Detected", (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(img, f"Position: {np.round(detection['position_camera'], 3)} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Angle: {round(detection['orientation_deg'], 1)}°", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                width_px = bottom_right[0] - top_left[0]
                height_px = bottom_right[1] - top_left[1]
                cv2.putText(img, f"Size: {width_px}x{height_px} px", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if 'depth' in detection:
                    try:
                        depth = detection['depth']
                        focal_length = detector.get_focal_length()
                        width_m = width_px * depth / focal_length
                        height_m = height_px * depth / focal_length
                        cv2.putText(img, f"Size: {round(width_m, 3)}x{round(height_m, 3)} m", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except Exception as e:
                        logging.warning("Depth-based size estimation failed: %s", e)

                cv2.imshow("Detection Window", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("User requested exit.")
                break
            elif key == ord('p'):
                logging.info("Paused by user.")
                cv2.waitKey(0)

    except NoObjectDetectedError as nde:
        logging.error("Object detection failure: %s", nde)
        print(nde)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        print("[INFO] Interrupted by user.")
    except Exception as ex:
        logging.exception("Unexpected error occurred:")
        print("[ERROR]", ex)
    finally:
        detector.release()
        cv2.destroyAllWindows()
        logging.info("Cleaned up resources.")

if __name__ == "__main__":
    main()


