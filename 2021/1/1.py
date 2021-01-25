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
    #blur = cv2.GaussianBlur(img, (31, 31), 0)
    threshold, thresh = cv2.threshold(bw, 20, 255, cv2.THRESH_BINARY)#+ cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    thresh = cv2.cvtColor(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
   # cv2.imshow("thresh", resize(thresh))
   # cv2.waitKey()
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
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    #thresh = cv2.dilate(thresh,kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Find better approx for connected components
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Copy all cc form original image
    #cv2.drawContours(blur, contours, -1, (0, 255, 0), 3)
    cv2.imshow("thresh first", cv2.cvtColor(resize(thresh), cv2.COLOR_GRAY2BGR))
    cv2.waitKey()
    '''
    border = 40
    height, width = bw.shape
    for contour in contours[1:]:
        x, y, w, h = cv2.boundingRect(contour)
        # Make it little bit bigger
        x, y = max(0, x - border), max(0, y - border)
        w = width - x if x + w + border > width else w + 2 * border
        h = height - y if y + h + border > height else h + 2 * border
        piece = blur[y:y + h, x:x + w]
        cv2.imshow("Piece", piece)
        cv2.waitKey()
        thresh_piece = cv2.threshold(piece, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh[y:y + h, x:x + w] = thresh_piece

    cv2.imshow("blur after", cv2.cvtColor(resize(thresh), cv2.COLOR_GRAY2BGR))
    cv2.waitKey()
    cv2.drawContours(bw, cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 3)
    cv2.imshow(f"balls count: {-1}", cv2.cvtColor(resize(bw), cv2.COLOR_GRAY2BGR))
    cv2.waitKey()
    
    img = cv2.erode(img, np.ones((10, 10), np.int8), iterations=2)
    img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    num_labels, labels_image = cv2.connectedComponents(img)

    # Map component labels to hue val
    label_hue = np.uint8(255 * labels_image / np.max(labels_image))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    cv2.imshow(f"balls count: {num_labels - 1}", labeled_img)
    cv2.waitkey()
    '''

    return {"count": len(radius_list), "mean": mean, "dispersion": d}

if __name__ == '__main__':
    print(count_balls(IMAGE_PATH))

