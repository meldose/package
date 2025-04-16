# main.py
import numpy as np # import numpy module
import cv2 # import cv2 module
from neurapy.robot import Robot
from object_detector import ObjectDetector # import object_detector
from robot_controller import RobotController # import robot_controller

def main(): # defining main function
    # Initialize detector & robot
    r=Robot()
    r.set_mode("Automatic") # setting the robot in automatic mode
    r.move_joint("New_capture") # setting the robot in New_capture position
    calibration_matrix = r"/home/hrg/Documents/package_detection/cam_to_tcp_transform.npy" # setting the calibration matrix
    #calibration_matrix = r"C:\Users\HeraldSuriaraj\Documents\neurapy-windows-v4.20.0\neurapy-windows-v4.20.0\cam_to_tcp_transform.npy"
    detector=ObjectDetector(calibration_matrix) # defining the object detector
    robot_control=RobotController() # defining the robot controller
    

    try:
        while True:
            detection=detector.get_detection() # getting the detection
            if detection is None: # if there is no detection
                print("[DEBUG] No object detected.")
                continue

            # clearly print detected position and orientation
            print("[DETECTED] Camera XYZ:", detection["position_camera"],"Angle deg:",detection["orientation_deg"])

            # Obtain current TCP pose
            tcp_pose_current=robot_control.robot.get_tcp_pose()

            T_tcp_to_base=robot_control.pose_to_matrix(tcp_pose_current,gripper_offset_z=-0.087) # adjusting the offset for the gripper

            pos_cam_hom=np.array([*detection["position_camera"],1]) # getting the camera position
            print(pos_cam_hom)
            base_coords=T_tcp_to_base @ detector.T_cam_to_tcp @ pos_cam_hom # getting the base coordinates
            print(base_coords)

            yaw_rad=np.deg2rad(detection["orientation_deg"]) # gettting the orientation angle over Z AXIS
            print(yaw_rad)

            target_pose=[base_coords[0],base_coords[1],base_coords[2],0,np.pi,yaw_rad] # defining the target pose
            print("[TARGET POSE]",target_pose)
            yaw_deg = np.rad2deg(yaw_rad) # setting the yaw angle
            print(yaw_deg)

            
            img=detection["color_image"] # setting the image
            cv2.circle(img, detection["pixel"],5,(0,255,0),-1) # getting the pixel
            cv2.imshow("Detection",img) # showing the image
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
            
            robot_control.move_to_pose(target_pose,speed=0.2) # moving to the target pose

            print("STARTING THE ROBOT")

            robot_control.move_robot_based_on_angle(yaw_rad=np.deg2rad(detection["orientation_deg"])) # MOVING THE ROBOT TO THE NEW DESIRED POSITION

            print("FINISHED ::::::::::::::::::::")

    except Exception as ex:
        print("[ERROR]",ex)
    finally:
        detector.release() # detecting the realsense camera
        cv2.destroyAllWindows() # destroying the windows

if __name__=="__main__":
    main() # calling the main function

