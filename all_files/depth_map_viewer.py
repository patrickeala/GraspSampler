
# from experiment_utils import utils
import pickle as pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

depth_maps = np.asarray(pickle.load(open(f'depth_maps/ycb/004_sugar_box.pkl','rb')))
print(depth_maps.shape)
print(depth_maps[0].shape)
plt.imshow(depth_maps[0],'gray')
plt.show()
# [504 747 237 202 548 248  31 758 752 802 407 875 725 285 341 831 801 938 265 
# 406 882 993 931 239 588 818 523 673 471 631 869 812 724 309 554 150525 218 
# 23 726 193 570 979 362 549 872 748 947 773 255]
