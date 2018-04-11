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


def process_image(im):
    ''' im is Image object, returned by call to open (Pillow) '''
    im_arr = np.asarray(im, dtype='float32')
    #im_arr = im_arr - im_arr.mean(axis=0).mean(axis=0)
    thresh = im_arr[:,:,0] > np.percentile(im_arr[:,:,0], 60)
    #diff = im_arr[thresh, 0] - im_arr[thresh, 1]
    corr, _ = pearsonr(im_arr[thresh, 0], im_arr[thresh, 1])
    return corr

healthy = os.listdir(os.path.join(BASE_DIR, HEALTH_DIR))
sick = os.listdir(os.path.join(BASE_DIR, SICK_DIR))
heal_avg_diff = []
sic_avg_diff = []
heal_sick_names =[]

for im in healthy:
    healthy_im = Image.open(os.path.join(BASE_DIR, HEALTH_DIR, im))
    diff_heal = process_image(healthy_im)
    heal_avg_diff.append(diff_heal)
    heal_sick_names.append(HEALTH_DIR + im)
    
for im in sick:
    sick_im = Image.open(os.path.join(BASE_DIR, SICK_DIR, im))
    diff_sic = process_image(sick_im)
    sic_avg_diff.append(diff_sic)
    heal_sick_names.append(SICK_DIR + im)

heal_avg = np.asarray(heal_avg_diff)
sic_avg = np.asarray(sic_avg_diff)
hs_names = np.asarray(heal_sick_names)

##

#np.random.shuffle(heal_avg)
#np.random.shuffle(sic_avg)



np.save("test_healthy", heal_avg)
np.save("test_sick", sic_avg)
np.save("test_names_hs", hs_names)
'''
fig1 = plt.figure()
ax11 = fig1.add_subplot(121)
ax11.imshow(thresh_heal, cmap='gray')
ax12 = fig1.add_subplot(122)
ax12.imshow(thresh_sic, cmap='gray')
fig1.suptitle("Threshold Masks")
plt.show()
'''
