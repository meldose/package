import numpy as np # importing numpy module
from neurapy.robot import Robot 
from scipy.spatial.transform import Rotation as R # importing rotation module
from object_detector import ObjectDetector # importing object_detector
import time # importing time module
class RobotController: # defining class RobotController
    def __init__(self, robot_ip=None): # intializing the class
        self.robot = Robot(robot_ip) if robot_ip else Robot()
        self.robot.set_mode("Automatic")

    def pose_to_matrix(self, tcp_pose, gripper_offset_z): # defining the function for pose to matrix
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
        T_tcp = np.eye(4) # creating a 4x4 identity matrix
        T_tcp[:3,:3] = rotation_matrix # setting the rotation matrix
        T_tcp[:3,3] = translation # setting the translation

        # Account explicitly for your gripper offset along the TCP's local Z-axis
        T_gripper_offset = np.eye(4)
        T_gripper_offset[:3, 3] = [0, 0, gripper_offset_z]  # Z offset of gripper w.r.t. TCP

        # Final transformation (base→gripper_tip)
        T_final = T_tcp @ T_gripper_offset

        return T_final # returning the final transformation

    def move_to_pose(self, pose_xyzrpy, speed=0.2): # defining a function named move_to_pose
        
        
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

        self.robot.move_linear_from_current_position(**linear_property) # move the robot to the given pose
        io_set = self.robot.set_tool_digital_outputs([1.0,0.0,0.0]) # setting the tool digital outputs
        time.sleep(1) # setting the sleep time
        self.robot.move_joint("New_capture") # move the robot to New_capture position
        self.robot.move_joint("P50") # move the robot to P50 position
        self.robot.move_joint("P57") # move the robot to P57 position

    def move_robot_based_on_angle(self,yaw_rad): # defining a function named move_robot_based_on_angle
        if yaw_rad <=1.2: # if the yaw angle is less than 1.2

            self.robot.move_joint([-1.1861648754996472,
            0.6636897458219015,
            0.8012450537926126,
            -0.0003520366065948493,
            1.6767200221419756,
            -1.6916913151049922]
            )
            
        elif yaw_rad <=0.015: # if the yaw angle is less than 0.015

            self.robot.move_joint([-1.1861626284574773,
            0.6636927418781279,
            0.8012458028066691,
            -0.0003550326628211884,
            1.6767226436911735,
            -2.830129014919057]
            )

        elif 0.01 >= yaw_rad <=0.07: # if the yaw angle is less than 0.015

            self.robot.move_joint([-1.1470438713242237,
            0.6759784449412598,
            0.8166148222337324,
            -0.00039622843593335163,
            1.6490518173987618,
            -1.03197134378299]
            )

        elif 1.3 >= yaw_rad <=1.5: # if the yaw angle is less than 0.015

            self.robot.move_joint([-1.1470438713242237,
            0.6759784449412598,
            0.8166148222337324,
            -0.00039622843593335163,
            1.6490518173987618,
            -1.03197134378299]
            )

        elif 0.2 >= yaw_rad <=0.4: # if the yaw angle is less than 0.015

            self.robot.move_joint([-1.18398861515824,
            0.596268369039507,
            0.8626878003753474,
            -0.00027001956739881525,
            1.6827125091016821,
            -0.5077053214959502]
            )


        elif 0.019 >= yaw_rad <=0.03: # if the yaw angle is less than 0.015

            self.robot.move_joint([-1.1861641264855907,
            0.6636957379343542,
            0.8012465518207258,
            -0.00035465815579289604,
            1.6767226436911735,
            -1.2494101244095532]

            )

        elif 0.4 >= yaw_rad <=0.9: # if the yaw angle is less than 0.9

            self.robot.move_joint([-1.1861648754996472,
            0.6636897458219015,
            0.8012450537926126,
            -0.0003520366065948493,
            1.6767200221419756,
            -1.6916913151049922]
            )

        elif yaw_rad >=1.6708: # if the yaw angle is greater than 1.6708

            self.robot.move_joint([0.18229699489025253,
            -0.45074154634338454,
            0.36958859562606833,
            -0.001521776648733053,
            0.6085402412020895,
            0.7935212814262113,
            0.0006595128791383861]
            )
        elif 1.52 >= yaw_rad <=1.59: # if the yaw angle is less than 1.59

            self.robot.move_joint([-1.1864124246453485,
                0.6649982733787552,
                0.8029329569691264,
                -0.00040783815381041584,
                1.6769492204432905,
                -1.555638154824762]
                )
            
        elif 1.57 >= yaw_rad <=1.58: # if the yaw angle is less than 1.59

            self.robot.move_joint([-1.1861678715558734,
            0.6726569421073346,
            0.7471812191883228,
            -0.0003531601276797265,
            1.7218234015873415,
            -2.6919392920424183]
                )
            
        elif 1.50 >= yaw_rad <=1.58: # if the yaw angle is less than 1.59

            self.robot.move_joint([-1.1861678715558734,
            0.6726569421073346,
            0.7471812191883228,
            -0.0003531601276797265,
            1.7218234015873415,
            -2.6919392920424183]
                )
            
        elif yaw_rad >=3.14159: # if the yaw angle is greater than 3.14159

            self.robot.move_joint([-1.1864124246453485,
                0.6649982733787552,
                0.8029329569691264,
                -0.00040783815381041584,
                1.6769492204432905,
                -1.555638154824762]
                )
            
        elif yaw_rad <=0.019: # if the yaw angle is equal to -3.14159

            self.robot.move_joint([-1.1864124246453485,
                0.6649982733787552,
                0.8029329569691264,
                -0.00040783815381041584,
                1.6769492204432905,
                -1.555638154824762]
                )
            
        elif 0.17 >= yaw_rad <=1.19: # if the yaw angle is equal to -3.14159

            self.robot.move_joint([-1.1865757097096838,
            0.6631530772503585,
            0.8013405530848271,
            -0.00040222054838602995,
            1.6769795555125822,
            -1.6692055386192886]
                )
            
        elif 0.17 >= yaw_rad <=0.6: # if the yaw angle is equal to -3.14159

            self.robot.move_joint([-1.1865757097096838,
            0.6631530772503585,
            0.8013405530848271,
            -0.00040222054838602995,
            1.6769795555125822,
            -1.6692055386192886]
                )
            
        elif 0.7 >= yaw_rad <=0.75: # if the yaw angle is equal to -3.14159

            self.robot.move_joint([-1.191919550496388,
            0.6405141273900834,
            0.932262218063395,
            -0.0001423126707511093,
            1.5689619913703203,
            0.1214376489940913]
            )

        elif 0.8 >= yaw_rad <=0.9: # if the yaw angle is equal to -3.14159

            self.robot.move_joint([-1.1533427050330736,
            0.7191894403726644,
            0.7156421098006789,
            -0.00015130083943012672,
            1.7069045396082858,
            -1.9343445304253974]
            )
            
            
            
        io_set = self.robot.set_tool_digital_outputs([0.0,1.0,0.0]) # setting the tool digital outputs
        self.robot.move_joint("New_capture") # setting the robot in New_capture position
        self.robot.stop() # stopping the robot