import pyrealsense2 as rs
import numpy as np
import cv2

# Chessboard parameters
pattern_size = (9, 6)
square_size = 0.025  # 25 mm squares

# Prepare object points (same for every image)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Load or define camera intrinsics
# You should calibrate the camera before this if needed
camera_matrix = np.array([[615.0, 0, 320.0],
                          [0, 615.0, 240.0],
                          [0, 0, 1]])
dist_coeffs = np.zeros(5)

# Create RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize storage
R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

# Example robot poses for each captured image
# Replace this with actual poses from your robot
# Each entry is (R_gripper2base, t_gripper2base)
# R: 3x3 numpy array, t: 3x1 numpy array
robot_poses = [...]  # <-- Fill in with your real data!

# Capture data
num_images = len(robot_poses)
captured = 0
print("Press SPACE to capture images when checkerboard is visible.")

try:
    while captured < num_images:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img = cv2.drawChessboardCorners(color_image.copy(), pattern_size, corners2, ret)
            cv2.imshow("Checkerboard", img)
        else:
            cv2.imshow("Checkerboard", color_image)

        key = cv2.waitKey(1)
        if key == 32 and ret:  # SPACE to capture
            print(f"Captured {captured+1}/{num_images}")

            # SolvePnP to get target-to-camera pose
            success, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            R_cam, _ = cv2.Rodrigues(rvec)

            R_target2cam.append(R_cam)
            t_target2cam.append(tvec)

            R_g, t_g = robot_poses[captured]
            R_gripper2base.append(R_g)
            t_gripper2base.append(t_g)

            captured += 1

        if key == 27:  # ESC to exit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Run hand-eye calibration
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("Camera to Gripper Rotation:\n", R_cam2gripper)
print("Camera to Gripper Translation:\n", t_cam2gripper)
