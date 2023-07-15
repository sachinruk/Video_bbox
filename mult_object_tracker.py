import cv2, os, sys
import argparse
from random import random

def disp_classes() -> str:
    temp_arr=[]
    temp_arr.append('{')
    for i in range(len(classes_)):
        temp_arr.append(f' {i}:{classes_[i]},')
    temp_arr.append('}')
    return ''.join(temp_arr)

# parse the arguments used to call this script
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='name of video file', type=str)
parser.add_argument('--max_obj', help='Maximum number of objects followed', type=int, default=20)
parser.add_argument('--max_frames', help='Maximum number of objects followed', type=int, default=500)
parser.add_argument('--thresh', help='Threshold for scene changes', type=float, default=2)
args = parser.parse_args()
max_obj = args.max_obj
max_frames = args.max_frames
thresh = args.thresh

fname =  os.path.basename(args.name)[:-4] #filename without extentsion
video = cv2.VideoCapture(args.name) # Read video

# create directories to store individual frames and their labels
label_path=os.path.join('\\'.join(args.name.split('\\')[:-1]),"labels")
img_path=os.path.join('\\'.join(args.name.split('\\')[:-1]),"images")
os.makedirs(label_path, exist_ok=True)
os.makedirs(img_path, exist_ok=True)
 
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

classes_=[]
classes_colors=[]

frames = 1
prev_mean = 0
while ok and frames <= max_frames:
    class_bbox=[]

    frame_diff = abs(frame.mean() - prev_mean)
    prev_mean = frame.mean()

    frame = cv2.resize(frame, (h, w))
    name = fname + '_' + str(frames).zfill(4)
    origFrame = frame.copy()
    bbox_frame  = frame.copy()
    
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key != ord("q") or frames == 1 or frame_diff > thresh:
        trackers = cv2.legacy.MultiTracker_create()
        for i in range(max_obj):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", bbox_frame, fromCenter=False)
            # create a new object tracker for the bounding box and add it
            # to our multi-object tracker
            if initBB[2] <= 1 or initBB[3] <= 1: # if no width or height
                break
            # # start OpenCV object tracker using the supplied bounding box
            tracker = cv2.legacy.TrackerCSRT_create()
            trackers.add(tracker, frame, initBB)
            
            while True:
                try:
                    class_bbox.append(int(input(f'{i+1}. Enter ClassNo.: {disp_classes()}: ')))
                except:
                    print('Enter a valid class no.')
                    continue
                if(class_bbox[-1]==len(classes_)):
                    classes_.append(input('Name of the object: '))
                    classes_colors.append((random()*255, random()*255, random()*255))
                    break
                elif(class_bbox[-1]>len(classes_)):
                    print('Enter a valid class no.')
                    continue
                else:
                    break
            
            bbox_frame = cv2.rectangle(bbox_frame, (initBB[0], initBB[1]), (initBB[0]+initBB[2], initBB[1]+initBB[3]), color=classes_colors[class_bbox[-1]], thickness=2)
            


    if key == ord("q"):
        break

    if initBB is not None:
        (tracking_ok, boxes) = trackers.update(frame)

        # save image and bounding box
        if tracking_ok:
            if len(boxes) > 0: # if there is a box that is being tracked
                cv2.imwrite(img_path + '/' + name + '.jpg', origFrame)
                with open(label_path + '/' + name + '.txt', 'a') as f:
                    for bbox,i in zip(boxes,class_bbox):
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                        centre = [0.5*(p1[1]+p2[1])/w, 0.5*(p1[0]+p2[0])/h]
                        width, height = (bbox[3]/w, bbox[2]/h)
                        f.write(f'{i} {centre[0]:.6f} {centre[1]:.6f} {width:.6f} {height:.6f}\n')
        else:
            initBB = None

    
    cv2.imshow("Frame", frame)

    ok,frame = video.read()
    frames += 1

video.release()
# close all windows
cv2.destroyAllWindows()
