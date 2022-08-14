import glob
import argparse
import os
from posixpath import split
import cv2
from scipy.fftpack import shift
from SIFT import SIFT
import numpy as np
import matplotlib.pyplot as plt
import random
from fitHomo import *
from warping import *
from cylindrical_projection import *
from matching import *

class Stitcher:
    def __init__(self):
        self.cache_kps = []
        self.cache_feature = []
        self.curr_shift = [[0,0]]
        self.warper = Warp()

    def stitch(self, stitched_img, img_l, img_r, blending_mode = "linearBlending"):
        print("feature detection")
        if len(self.cache_feature)==0 or len(self.cache_kps)==0:
            img_l = img_l.astype('uint8')
            img_l_kps, img_l_features = self.featureDetection(img_l)
            #img_l_kps, img_l_features = SIFT(img_l)
            #img_l = cv2.drawKeypoints(img_l, img_l_kps, img_l, color=(255,0,0))
            #cv2.imwrite('key_point_1.png', img_l)
        else:
            img_l_kps, img_l_features = self.cache_kps, self.cache_feature

        img_r = img_r.astype('uint8')
        img_r_kps, img_r_features = self.featureDetection(img_r)
        #img_r_kps, img_r_features = SIFT(img_r)
        #img_r = cv2.drawKeypoints(img_r, img_r_kps, img_r, color=(255,0,0))
        #cv2.imwrite('key_point_2.png', img_r)

        print("feature matching")
        '''
        =================================Here!!!!!!!!!!!!!!!!!!!!!=========================
        '''
        matches_pos = matchKeyPoint(img_l_kps, img_r_kps, img_l_features, img_r_features, ratio = 0.75)
        drawMatches([img_l, img_r], matches_pos)

        #fit the homography model with RANSAC algorithm
        print("RANSAC")
        shift = RANSAC(matches_pos)
        self.curr_shift.append(shift[0])
        #HomoMat = fitHomoMat(matches_pos)
        #print(HomoMat)
        warp_img = self.warper.warp([stitched_img, img_r], self.curr_shift)
        #warp_img = warp([stitched_img, img_r], HomoMat, blending_mode)
        

        self.cache_kps = img_r_kps
        self.cache_feature = img_r_features
        
        return warp_img

    def featureDetection(self, img):
        sift = cv2.SIFT_create()
        kps, features = sift.detectAndCompute(img,None)

        return kps, features



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, default='./data/scene_1')

    args, unknown = parser.parse_known_args()

    stitcher = Stitcher()

    #Read images
    if 'parrington' in args.img_dir:
        images = sorted(glob.glob(os.path.join(args.img_dir, 'prtn*.jpg')))
        #Read focal length
        focal = []
        f = open(os.path.join(args.img_dir, 'pano.txt'))
        for line in f.readlines():
            line = line.split(' ')
            if len(line) == 1 and line[0]!='\n'and 'jpg' not in line[0]:
                focal.append(float(line[0].split('\n')[0]))
    if 'scene_1' in args.img_dir:
        images = sorted(glob.glob(os.path.join(args.img_dir, 'DSC*.jpg')))
        #Read focal length
        focal = []
        f = open(os.path.join(args.img_dir, 'pano.txt'))
        for line in f.readlines():
            line = line.split(' ')
            if len(line) == 1 and line[0]!='\n'and 'jpg' not in line[0]:
                focal.append(float(line[0].split('\n')[0]))
    if 'scene_8' in args.img_dir:
        images = sorted(glob.glob(os.path.join(args.img_dir, 'DSC*.JPG')))
        #Read focal length
        focal = []
        f = open(os.path.join(args.img_dir, 'pano.txt'))
        for line in f.readlines():
            line = line.split(' ')
            if len(line) == 1 and line[0]!='\n'and 'JPG' not in line[0]:
                focal.append(float(line[0].split('\n')[0]))

    print(focal)


    cylindrical_images = []
    for i , filename in enumerate(images):
        img = cv2.imread(filename)
        H, W, _ = img.shape
        img = cv2.resize(img, (W//8, H//8), interpolation=cv2.INTER_AREA)
        cylindrical_images.append(cylindrical_projection(img, focal[i]))

    stitched_img = cylindrical_images[0].copy()

    for i ,(img1, img2) in enumerate(zip(cylindrical_images[:-1], cylindrical_images[1:])):
        #img1 = cv2.imread(filename_L, cv2.IMREAD_GRAYSCALE)  #(H, W, 3)
        #img2 = cv2.imread(filename_R, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite('stitch_input.png', img2)
        stitched_img = stitcher.stitch(stitched_img, img1, img2)
        cv2.imwrite('Stich_result.png', stitched_img.astype('uint8'))
        input("finish")
           
