import cv2
import numpy as np

# Lists to store rotation and translation vectors
R_gripper2base = []  # from robot
t_gripper2base = []
R_target2cam = []    # from solvePnP
t_target2cam = []
 
num_samples=10
# Example: loop over multiple images and robot poses
for i in range(num_samples):
    # Load image and detect chessboard
    image="/home/hrg/Desktop/package/checkerboard_9x6_25mm.png"
    cols=9
    rows=6
    ret, corners = cv2.findChessboardCorners(image, (cols, rows))
    if ret:
        # Solve PnP
        ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
        R_cam, _ = cv2.Rodrigues(rvec)
        
        # Append target to cam
        R_target2cam.append(R_cam)
        t_target2cam.append(tvec)

        # Append robot pose (gripper to base) for this image
        R_g = ...  # 3x3 rotation from robot
        t_g = ...  # 3x1 translation from robot
        R_gripper2base.append(R_g)
        t_gripper2base.append(t_g)

# Run calibration
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("Rotation:\n", R_cam2gripper)
print("Translation:\n", t_cam2gripper)
