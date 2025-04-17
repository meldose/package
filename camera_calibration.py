import cv2
import numpy as np
import glob

# Set the chessboard size (number of inner corners per chessboard row and column)
chessboard_size = (9, 6)  # (columns, rows) of internal corners
square_size = 1.0  # Set this to the real size of your squares (e.g., 25mm, 1cm, etc.)

# Prepare 3D points in real world space (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Load calibration images
images = glob.glob('calibration_images/*.jpg')  # adjust path and extension

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Print the results
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    # Save to file
    np.savez("calibration_data.npz", 
            camera_matrix=camera_matrix, 
            dist_coeffs=dist_coeffs, 
            rvecs=rvecs, 
            tvecs=tvecs)
