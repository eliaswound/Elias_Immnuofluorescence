import os
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Data PATH
BASE_DIR = '/home/yasaman/Documents/Winter18/fluorescence/'
HEALTH_DIR ='healthy/'
SICK_DIR = 'sick/'


def process_image(im):
    ''' im is Image object, returned by call to open (Pillow) '''
    im_arr = np.asarray(im, dtype='float32')
    im_arr = im_arr - im_arr.mean()
    thresh= im_arr[:,:,0] > np.percentile(im_arr[:,:,0], 80)
    diff = im_arr[thresh, 0] - im_arr[thresh, 1]
    return diff

healthy = os.listdir(os.path.join(BASE_DIR, HEALTH_DIR))
sick = os.listdir(os.path.join(BASE_DIR, SICK_DIR))

heal_avg_diff = []
sic_avg_diff = []

for im in healthy:
    healthy_im = Image.open(os.path.join(BASE_DIR, HEALTH_DIR, im))
    diff_heal = process_image(healthy_im)
    heal_avg_diff.append(diff_heal.mean())
    
for im in sick:
    sick_im = Image.open(os.path.join(BASE_DIR, SICK_DIR, im))
    diff_sic = process_image(sick_im)
    sic_avg_diff.append(diff_sic.mean())

heal_avg = np.asarray(heal_avg_diff)
sic_avg = np.asarray(sic_avg_diff)


##
# shuffle and set aside a test set
test_ratio = 0.2
heal_test_num = int((test_ratio * heal_avg.shape[0])//1)
sic_test_num = int((test_ratio * sic_avg.shape[0])//1)

np.random.shuffle(heal_avg)
np.random.shuffle(sic_avg)

test_healthy = heal_avg[:heal_test_num]
train_healthy = heal_avg[heal_test_num:]
test_sick = sic_avg[:sic_test_num]
train_sick = sic_avg[sic_test_num:]


np.save("test_healthy", test_healthy)
np.save("train_healthy", train_healthy)
np.save("test_sick", test_sick)
np.save("train_sick", train_sick)

'''
fig1 = plt.figure()
ax11 = fig1.add_subplot(121)
ax11.imshow(thresh_heal, cmap='gray')
ax12 = fig1.add_subplot(122)
ax12.imshow(thresh_sic, cmap='gray')
fig1.suptitle("Threshold Masks")
plt.show()
'''
