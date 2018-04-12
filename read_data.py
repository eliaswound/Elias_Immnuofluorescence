import os
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   

# Data PATH
BASE_DIR = '/home/yasaman/Mennella_Lab/SPINNING_DISC_1_23_2018/'
HEALTH_DIR ='healthy21/'
SICK_DIR = 'sick21/'

# test data path
TEST_BASE = '/home/yasaman/Mennella_Lab/spin_disc2/'
TEST_HEALTH = ['healthy_3_28', 'healthy_3_26']
TEST_SICK = ['pcd22_04_03', 'pcd22_3_30']

def process_image(im):
	''' im is Image object, returned by call to open (Pillow) '''
	im_arr = np.asarray(im, dtype='float32')
	thresh = im_arr[:,:,0] > np.percentile(im_arr[:,:,0], 20)
	corr, _ = pearsonr(im_arr[thresh, 0], im_arr[thresh, 1])
	return corr


def get_corr_im_dir(BASE_DIR, CLASS_DIR):	
	im_list = os.listdir(os.path.join(BASE_DIR, CLASS_DIR))
	corr_list = []
	names = []
	for im in im_list:
		im_ob = Image.open(os.path.join(BASE_DIR, CLASS_DIR, im))
		corr = process_image(im_ob)
		corr_list.append(corr)
		names.append(CLASS_DIR + im)

	return np.asarray(corr_list), np.asarray(names)




if (__name__ == "__main__"):
	train_healthy, train_healthy_names = get_corr_im_dir(BASE_DIR, HEALTH_DIR)
	train_sick, train_sick_names = get_corr_im_dir(BASE_DIR, SICK_DIR)
	
	test_healthy = []
	test_healthy_names = []
	test_sick = []
	test_sick_names = []
	for dirc in TEST_HEALTH:
		corrs, names = get_corr_im_dir(TEST_BASE, dirc)
		test_healthy.append(corrs)
		test_healthy_names.append(names)
	test_healthy = np.concatenate(test_healthy, axis=0)
	test_healthy_names = np.concatenate(test_healthy_names, axis=0)
	
	for dirc in TEST_SICK:
		corrs, names = get_corr_im_dir(TEST_BASE, dirc)
		test_sick.append(corrs)
		test_sick_names.append(names)
	test_sick = np.concatenate(test_sick, axis=0)
	test_sick_names = np.concatenate(test_sick_names, axis=0)
		



	np.save("train_healthy", train_healthy)
	np.save("train_sick", train_sick)

	np.save("test_healthy", test_healthy)
	np.save("test_sick", test_sick)

		



