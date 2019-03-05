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
	wt_17 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/WT_PCD17'
	wt_23 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/WT_PCD23'
	wt_40E = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/WT_PCD40E'
	wt_40Z = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/WT_PCD40Z'
	pcd36 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/PCD36_cropped'
	pcd17 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/PCD17_cropped'
	pcd23 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/PCD23_cropped'
	pcd40 = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/PCD40_cropped'
	
	wt_cor, _ = get_corr_im_dir(test_wt)
	wt17_cor, _ = get_corr_im_dir(wt_17)
	wt23_cor, _ = get_corr_im_dir(wt_23)
	wt40E_cor, _ = get_corr_im_dir(wt_40E)
	wt40Z_cor, _ = get_corr_im_dir(wt_40Z)
	pcd36_cor, _ = get_corr_im_dir(pcd36)
	pcd17_cor, _ = get_corr_im_dir(pcd17)
	pcd23_cor, _ = get_corr_im_dir(pcd23)
	pcd40_cor, _ = get_corr_im_dir(pcd40)



	np.save("wt", wt_cor)
	np.save("wt17", wt17_cor)
	np.save("wt23", wt23_cor)
	np.save("wt40E", wt40E_cor)
	np.save("wt40Z", wt40Z_cor)
	np.save("pcd17", pcd17_cor)
	np.save("pcd36", pcd36_cor)
	np.save("pcd23", pcd23_cor)
	np.save("pcd40", pcd40_cor)

	
		



