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
    img = cv2.imread('GOPR01170000.jpg')
    # Get params from opencv storage
    storage_filename = "camera.xml"
    s = cv2.FileStorage()
    s.open(storage_filename, cv2.FileStorage_READ)
    camera_matrix = s.getNode('camera_matrix').mat()
    dist_coeffs = s.getNode('distortion_coefficients').mat()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    cv2.imwrite('result.jpg', dst)


def task_three():
    """
    """
if __name__ == '__main__':
    tasks = [task_one, task_two, task_three]
    for i, task in enumerate(tasks):
        print("-" * 20 + " Task ", i + 1, "-" * 20)
        task()