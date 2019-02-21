import cv2, os, sys
import argparse

# create directories to store individual frames and their labels
os.makedirs("./labels", exist_ok=True)
os.makedirs("./images", exist_ok=True)

# parse the arguments used to call this script
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='name of video file', type=str)
parser.add_argument('--max_obj', help='Maximum number of objects followed', type=int, default=6)
parser.add_argument('--thresh', help='Threshold for scene changes', type=float, default=2)
args = parser.parse_args()
max_obj = args.max_obj
thresh = args.thresh

fname =  os.path.basename(args.name)[:-4] #filename without extentsion
video = cv2.VideoCapture(args.name) # Read video
 
# Exit if video not opened
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame
ok,frame = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

# h, w, _ = frame.shape
# import pdb; pdb.set_trace()
h = w = 608
initBB = None

frames = 1
prev_mean = 0
while ok:
    frame_diff = abs(frame.mean() - prev_mean)
    prev_mean = frame.mean()

    frame = cv2.resize(frame, (h, w))
    name = fname + '_' + str(frames).zfill(4)
    cv2.imwrite('./images/' + name + '.jpg', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s") or frames == 1 or frame_diff > thresh:
        trackers = cv2.MultiTracker_create()
        for i in range(max_obj):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False) 
            # create a new object tracker for the bounding box and add it
            # to our multi-object tracker
            if initBB[2] == 0 or initBB[3] == 0: # if no width or height
                break
            tracker = cv2.TrackerCSRT_create()
            trackers.add(tracker, frame, initBB)

            # # start OpenCV object tracker using the supplied bounding box
            # # coordinates, then start the FPS throughput estimator as well
            # tracker.init(frame, initBB)
    elif key == ord("q"):
        break

    if initBB is not None:
        (tracking_ok, boxes) = trackers.update(frame)

        # Draw bounding box
        if tracking_ok:
            with open('./labels/' + name + '.txt', 'a') as f:
                for bbox in boxes:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                    centre = [0.5*(p1[1]+p2[1])/w, 0.5*(p1[0]+p2[0])/h]
                    width, height = (bbox[3]/w, bbox[2]/h)
                    f.write(f'0 {centre[0]:.6f} {centre[1]:.6f} {width:.6f} {height:.6f}\n')
        else:
            initBB = None

    
    cv2.imshow("Frame", frame)

    ok,frame = video.read()
    frames += 1

video.release()
# close all windows
cv2.destroyAllWindows()