import cv2
import numpy as np

IMAGE_PATH = "whiteballssample.jpg"

def resize(img, scale_percent=30):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def count_balls(image_path):
    # Read image
    img = cv2.imread(image_path)
    # Make it b&w
    b = img[:, :, 2]
    bw = cv2.merge([b]*3)
    bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
    cv2.imshow("blue", resize(bw))
    cv2.waitKey()
    # Make binarization
    threshold, thresh = cv2.threshold(bw, 20, 255, cv2.THRESH_BINARY)#+ cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    thresh = cv2.cvtColor(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 2, 40,
                              param1=200, param2=40, minRadius=0, maxRadius=100)
    assert circles is not None
    circles = np.uint16(np.around(circles))
    radius_list = circles[0, :, 2]
    mean = radius_list.mean()
    d = ((radius_list - mean)**2).mean()
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow(f"Detected circles: count = {len(radius_list)}, mean = {mean}, dispersion = {d}", resize(img))
    cv2.waitKey()

    return {"count": len(radius_list), "mean": mean, "dispersion": d}

if __name__ == '__main__':
    print(count_balls(IMAGE_PATH))

