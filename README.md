# Bounding Box Generator for Videos

This tool takes a video and allows you to put bounding boxes on the frames, without having to do it on individual frames. It will automatically stop when scenes/ camera angels change prompting you for bounding boxes again.

This tool OpenCV's CSRT tracker to achieve this.

## Usage
`python mult_object_tracker.py --name /path/to/video.file` which will create the labels in the YOLO format (class_id, relative_center, relative_width, relative height). Currently class_id is simply 0 since it can only look for one class. The images will be stored as a `jpg` in folder `images` and corresponding labels in folder named `labels`.

It will prompt you for bounding boxes in first frame. Draw the box and press enter. You may redraw if you made a mistake as long as you don't press enter. By default you can draw upto 6 boxes. But this can be changed with the `--max_obj` option.

### Notes
- If you have less than the maximum number of objects, simply press `enter` after you have drawn the required number of boxes.
- If the bounding boxes are of incorrect shapes, press `s` to stop the video and redraw.
- If you have finished drawing boxes, press `q`.

## Requirements
- OpenCV 3+
- Python 3.5+