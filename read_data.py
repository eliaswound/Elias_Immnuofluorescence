import os
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   
from scipy.ndimage.filters import gaussian_filter


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


	test_wt = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/WT_cropped'
	pcd35 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/PCD35_cropped'
	pcd36 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/PCD36_cropped'
	pcd38 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/PCD38_cropped'
	
	wt_cor, _ = get_corr_im_dir(test_wt)
	pcd35_cor, _ = get_corr_im_dir(pcd35)
	pcd36_cor, _ = get_corr_im_dir(pcd36)
	pcd38_cor, _ = get_corr_im_dir(pcd38)



	np.save("wt", wt_cor)
	np.save("pcd35", pcd35_cor)
	np.save("pcd36", pcd36_cor)
	np.save("pcd38", pcd38_cor)

	
		



