# Remember to execute under "hw2_[45]/code" folder!
# cd D:\user\Documents\Alex\Code\2022_VFX\hw2_[45]\code

# generic libraries
import glob
import math
import numpy as np
from matplotlib import pyplot as plt
import cv2

# customize libraries
import feature_matching
import image_matching

if __name__=='__main__':
    # read images & convert to gray images
    images = [cv2.imread(file) for file in glob.glob("../data/original_image/scene_1/*.jpg")]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # feature detection
    kp = []
    des = []
    sift = cv2.SIFT_create()
    for i in range(len(gray_images)):
        k, d = sift.detectAndCompute(gray_images[i], None)
        kp.append(k)
        des.append(d)
    print(type(des[0][0]))

    # feature matching
    matches = feature_matching.match(des[0], des[1])
    print(matches[0].queryIdx)
    print(matches[0].trainIdx)
    print(matches[0].distance)
