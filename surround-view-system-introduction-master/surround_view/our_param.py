import os
import cv2

yamldir = "images/0703/yaml"
project_imgpath = "images/0703/images-up"
project_imgtype = ".jpg"

weight_imgpath = "images/old/around"
weight_imgtype = ".png"

camera_names = ["front", "back", "left", "right"]

# --------------------------------------------------------------------
# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
# 在水平和垂直方向上，鸟瞰图看起来离校准模式有多远
shift_w = 200
shift_h = 200

# size of the gap between the calibration pattern and the car
# in horizontal and vertical directions
# 校准图案与小车在水平和垂直方向上的间隙大小
# inn_shift_w = 50
# inn_shift_h = 37
inn_shift_w = 0
inn_shift_h = 0

# total width/height of the stitched image
# 標定布大小 300*500 #50x2 #37x2
total_w = 300 + 2 * shift_w
total_h = 500 + 2 * shift_h

# four corners of the rectangular region occupied by the car
# top-left (x_left, y_top), bottom-right (x_right, y_bottom)
# 汽车所占据的矩形区域的四个角
xl = shift_w + 80 + inn_shift_w
xr = total_w - xl
yt = shift_h + 90 + inn_shift_h
yb = total_h - yt
# --------------------------------------------------------------------

project_shapes = {
    "front": (total_w, yt),
    "back": (total_w, yt),
    "left": (total_h, xl),
    "right": (total_h, xl)
}

# pixel locations of the four points to be chosen.
# you must click these pixels in the same order when running
# the get_projection_map.py script
project_keypoints = {
    "front": [(shift_w + 60, shift_h),  # scale o.6x0.8
              (shift_w + 240, shift_h),
              (shift_w + 60, shift_h + 80),
              (shift_w + 240, shift_h + 80)],

    "back": [(shift_w + 60, shift_h),
             (shift_w + 240, shift_h),
             (shift_w + 60, shift_h + 80),
             (shift_w + 240, shift_h + 80)],

    "left": [(shift_h + 100, shift_w),  # -scale 0.7 0.8 -shift -10 -20
             (shift_h + 400, shift_w),
             (shift_h + 100, shift_w + 60),
             (shift_h + 400, shift_w + 60)],

    "right": [(shift_h + 100, shift_w),
              (shift_h + 400, shift_w),
              (shift_h + 100, shift_w + 60),
              (shift_h + 400, shift_w + 60)]
}

car_image = cv2.imread(os.path.join(os.getcwd(), "images", "car.png"))
car_image = cv2.resize(car_image, (xr - xl, yb - yt))
