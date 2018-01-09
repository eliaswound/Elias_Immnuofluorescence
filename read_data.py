import os
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Data PATH
BASE_DIR = '/home/yasaman/Documents/Winter18/fluorescence/'
HEALTH_DIR ='healthy/'
SICK_DIR = 'sick/'

healthy = os.listdir(os.path.join(BASE_DIR, HEALTH_DIR))
sick = os.listdir(os.path.join(BASE_DIR, SICK_DIR))

heal_avg_diff = []
sic_avg_diff = []

for im in healthy:
    healthy_im = Image.open(os.path.join(BASE_DIR, HEALTH_DIR, im))
    heal = np.asarray(healthy_im, dtype='float32')
    thresh_heal = heal[:,:,0] > np.percentile(heal[:,:,0], 80)
    diff_heal = heal[thresh_heal,0] - heal[thresh_heal,1]
    heal_avg_diff.append(diff_heal.mean())
    
for im in sick:
    sick_im = Image.open(os.path.join(BASE_DIR, SICK_DIR, im))
    sic = np.asarray(sick_im, dtype='float32')
    thresh_sic = sic[:,:,0] > np.percentile(sic[:,:,0], 80)
    diff_sic = sic[thresh_sic,0] - sic[thresh_sic,1]
    sic_avg_diff.append(diff_sic.mean())

heal_avg = np.asarray(heal_avg_diff)
sic_avg = np.asarray(sic_avg_diff)

##
# shuffle and set aside a test set
test_ratio = 0.2
heal_test_num = (test_ratio * heal_avg.shape[0])//1
sic_test_num = (test_ratio * sic_avg.shape[0])//1

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
