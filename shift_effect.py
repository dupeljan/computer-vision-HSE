import cv2
import numpy as np

def main():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    th, tw = 0, 0
    speedh, speedw = 6, 6
    w, h = vid.read()[1].shape[:-1]
    out = cv2.VideoWriter('outpy_2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (h, w))


    while(True):
        th = th + speedh if th + speedh < h else 0
        tw = tw + speedw if tw + speedw < w else 0

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
        # Coloring
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.threshold(new_frame, 127, 255, cv2.THRESH_BINARY)[1]
        # Display the resulting frame
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_GRAY2BGR)
        out.write(new_frame)
        cv2.imshow('frame', new_frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    out.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
