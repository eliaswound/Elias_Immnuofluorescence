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


def get_corr_im_dir(DIR):	
	im_list = os.listdir(DIR)
	corr_list = []
	names = []
	for im in im_list:
		im_ob = Image.open(os.path.join(DIR, im))
		corr = process_image(im_ob)
		corr_list.append(corr)
		names.append(DIR + im)

	return np.asarray(corr_list), np.asarray(names)




if (__name__ == "__main__"):
	spin_sick = ['/home/yasaman/Mennella_Lab/SPINNING_DISC_1_23_2018/sick21/',
		'/home/yasaman/Mennella_Lab/spin_disc2/pcd22_04_03',
		'/home/yasaman/Mennella_Lab/spin_disc2/pcd22_3_30']
	spin_healthy = ['/home/yasaman/Mennella_Lab/spin_disc2/healthy_3_26',
		'/home/yasaman/Mennella_Lab/spin_disc2/healthy_3_28',
		'/home/yasaman/Mennella_Lab/SPINNING_DISC_1_23_2018/sick21/']

	wide_sick = '/home/yasaman/Mennella_Lab/WIDEFIELD_08_01_2018/PATIENT'
	wide_healthy ='/home/yasaman/Mennella_Lab/WIDEFIELD_08_01_2018/HEALTHY'

	wide_sick_cor, _ = get_corr_im_dir(wide_sick)
	wide_heal_cor, _ = get_corr_im_dir(wide_healthy)
	
	spin_sick_cor = []
	spin_heal_cor = []
	for dirc in spin_sick:
		corrs, _ = get_corr_im_dir(dirc)
		spin_sick_cor.append(corrs)
	spin_sick_cor = np.concatenate(spin_sick_cor, axis=0)
	
	for dirc in spin_healthy:
		corrs, _ = get_corr_im_dir(dirc)
		spin_heal_cor.append(corrs)
	spin_heal_cor = np.concatenate(spin_heal_cor, axis=0)
		



	np.save("spin_healthy", spin_heal_cor)
	np.save("spin_sick", spin_sick_cor)

	np.save("wide_healthy", wide_heal_cor)
	np.save("wide_sick", wide_sick_cor)

		



