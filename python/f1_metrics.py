import sys
import os
import numpy as np
import cv2

sequence_path = sys.argv[1]
mask_path = sys.argv[2]

sequence_filenames = os.listdir(sequence_path)
sequence_filenames.sort()
for imagename in sequence_filenames:
	image  = cv2.imread(os.path.join(sequence_path, imagename), 0)
	mask   = cv2.imread(os.path.join(mask_path, imagename), 0)
	mask[mask==0] = 255
	mask[mask<255]  =  0
	res = cv2.bitwise_and(image,mask)
	cv2.imshow('KITTI', res)
	cv2.waitKey(25)