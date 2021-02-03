import cv2
import numpy as np
from utils import resize


def sigma(x):
    return 1 / (1 + np.exp(-x))


def adaptive_conv(integral, y_equalize, dst, k = 9):
    filtered = np.empty_like(integral)
    h, w = integral.shape
    for i in range(integral.shape[0]):
        for j in range(integral.shape[1]):
            size = int(k * sigma(1 - dst[i, j]))
            if not size:
                filtered[i, j] = y_equalize[i, j]
                continue
            x0, y0 = max(i - size, 0), max(j - size, 0)
            x1, y1 = min(i + size, h - 1), min(j + size, w - 1)
            filtered[i, j] = integral[x0, y0] + integral[x1, y1] \
                             - integral[x1, y0] - integral[x0, y1]
            filtered[i, j] /= (2 * size) ** 2
            filtered[i, j] = min(filtered[i, j], 255)
    return filtered.astype(np.uint8)


def task(input_path):
    img = cv2.imread(input_path)
    print("Computing... Please wait calmly about two minutes..")
    #cv2.imshow("Original image. Computing... Please wait...", resize(img, 30))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = cv2.cvtColor(cv2.merge([img[:, :, 0]]*3), cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Y", resize(y, 20))
    #cv2.waitKey()
    y_equalize = cv2.equalizeHist(y)
    #cv2.imshow("Equalized", resize(y_equalize, 20))
    #cv2.waitKey()
    Canny = cv2.Canny(y_equalize, 100, 200)
    #cv2.imshow("Canny", resize(Canny, 20))
    #cv2.waitKey()
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
    #cv2.imshow('Corners', resize(Canny, 20))
    #cv2.waitKey()
    thresh = cv2.threshold(Canny, 20, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #cv2.imshow("Thresh + dilate", resize(thresh, 20))
    #cv2.waitKey()
    dst = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    cv2.normalize(dst, dst, 0, 1.0, cv2.NORM_MINMAX)
    #cv2.imshow("Distance map", resize(dst, 20))
    #cv2.waitKey()
    integral = cv2.integral(y_equalize)[1:, 1:]
    # Filtering
    filtered = adaptive_conv(integral, y_equalize, dst)
    #cv2.imshow("Filtered: ", resize(filtered, 30))
    #cv2.waitKey()
    img[:, :, 0] = filtered
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    print("Complete")
    cv2.imshow("Result", resize(img, 20))
    cv2.waitKey()
    cv2.imwrite("out.jpg", img)
    pass


if __name__ == '__main__':
    task("inp.jpg")