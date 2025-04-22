import cv2
import numpy as np
import glob

# Chessboard config
chessboard_size = (6, 9)  # 6x9 corners
square_size = 0.025       # 2.5 cm per square

# Prepare object points (0,0,0), (0.025,0,0), ..., (0.125,0.1,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2)
objp *= square_size

# Lists for data
R_target2cam = []
t_target2cam = []
R_gripper2base = []
t_gripper2base = []

# Simulated robot poses (replace with real robot API calls)
def get_simulated_robot_pose(i):
    angle = i * 0.05
    R = cv2.Rodrigues(np.array([angle, 0, 0]))[0]
    t = np.array([[0.1 * np.sin(angle)], [0.0], [0.2 + 0.01 * i]])
    return R, t

# Load chessboard images
images = sorted(glob.glob("images/*.jpg"))  # You need to capture or save calibration images here

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # Get camera pose relative to chessboard
        ret, rvec, tvec = cv2.solvePnP(objp, corners_sub, np.eye(3), None)  # Replace np.eye(3) with actual camera intrinsics
        R_cam, _ = cv2.Rodrigues(rvec)

        # Store inverse (chessboard to camera)
        R_target2cam.append(R_cam)
        t_target2cam.append(tvec)

        # Simulate robot motion (or get from robot)
        R_grip, t_grip = get_simulated_robot_pose(i)
        R_gripper2base.append(R_grip)
        t_gripper2base.append(t_grip)

        # Visualize
        cv2.drawChessboardCorners(img, chessboard_size, corners_sub, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Perform hand-eye calibration
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("Rotation (camera to gripper):\n", R_cam2gripper)
print("Translation (camera to gripper):\n", t_cam2gripper)

# Compose transformation matrix
T_cam2gripper = np.eye(4)
T_cam2gripper[:3, :3] = R_cam2gripper
T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
print("\nHomogeneous Transformation Matrix (Camera to Gripper):\n", T_cam2gripper)
