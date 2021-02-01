import cv2
import numpy as np
from utils import resize

def task(input_path):
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = cv2.cvtColor(cv2.merge([img[:, :, 0]]*3), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Y", resize(y, 20))
    cv2.waitKey()
    y_equalize = cv2.equalizeHist(y)
    cv2.imshow("Equalized", resize(y_equalize, 20))
    cv2.waitKey()
    Canny = cv2.Canny(y_equalize, 100, 200)
    cv2.imshow("Canny", resize(Canny, 20))
    cv2.waitKey()
    # Find corners
    dst = cv2.cornerHarris(Canny, 5, 3, 4e-3)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    thresh = int(np.quantile(dst_norm, 1 - 250/np.dot(*dst_norm.shape)))

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                cv2.circle(Canny, (j, i), 10, 255, -1)
    cv2.imshow('Corners', resize(Canny, 20))
    cv2.waitKey()
    thresh = cv2.threshold(Canny, 20, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Thresh", cv2.resize(thresh, 20))
    cv2.waitKey()
    dst = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    cv2.imshow("Distance", resize(dst, 20))
    cv2.waitKey()
    pass


if __name__ == '__main__':
    task("inp.jpg")