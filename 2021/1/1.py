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
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Make binarization
    blur = cv2.GaussianBlur(img, (31, 31), 0)
    threshold, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    # Find better approx for connected components
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Copy all cc form original image
    #cv2.drawContours(blur, contours, -1, (0, 255, 0), 3)
    cv2.imshow("thresh first", cv2.cvtColor(resize(blur), cv2.COLOR_GRAY2BGR))
    cv2.waitKey()
    border = 40
    height, width= img.shape
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
    cv2.drawContours(img, cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 3)
    cv2.imshow(f"balls count: {-1}", cv2.cvtColor(resize(img), cv2.COLOR_GRAY2BGR))
    cv2.waitKey()
    '''
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

    return num_labels - 1

if __name__ == '__main__':
    count_balls(IMAGE_PATH)

