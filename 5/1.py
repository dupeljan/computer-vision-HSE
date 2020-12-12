# 3D Computer Vision Course HSE Mag Data Mining 20
# Third exercise
# Daniil Lyakhov
# dupeljan@gmail.com
# dalyakhov@edu.hse.ru
import PIL.Image
import PIL.ExifTags
import numpy as np
import cv2

def task_one():
    '''Find intrinsic camera parametrs
    matrix by given
    image under the assumption that
    the optical axis goes strictly through
    the center of the image. Focal
    length is 24mm'''
    # Get camera params from EXIF
    img = PIL.Image.open('GOPR01170000.jpg')
    exif = {PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS}

    # Compute camera params
    mm_per_inch = 25.4
    focal_length = 24
    cx, cy = [exif[val] // 2 for val in ["ExifImageWidth", "ExifImageHeight"]]
    fx, fy = [focal_length * exif[val] * mm_per_inch for val in ["XResolution", "YResolution"]]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    print("K= \n", K)
    return K

def task_two():
    """Use cv2.undistort"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    img = cv2.imread("GOPR01170000.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

if __name__ == '__main__':
    task_two()
