import cv2
import numpy as np

def main():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    th, tw = 0, 0
    speedh, speedw = 10, 10
    w, h = vid.read()[1].shape[:-1]
    while(True):
        th = th + speedh if th < h else 0
        tw = tw + speedw if tw < w else 0

        shifth = int(th)
        shiftw = int(tw)
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        new_frame = np.empty_like(frame)
        new_frame[:, :(h - shifth), :], new_frame[:, (h - shifth):, :] = \
                                                    frame[:, shifth:, :], frame[:, :shifth, :]
        new_frame[:(w - shiftw), :, :], new_frame[(w - shiftw):, :, :] = \
            new_frame[shiftw:, :, :], new_frame[:shiftw, :, :].copy()
        # Display the resulting frame
        cv2.imshow('frame', new_frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
