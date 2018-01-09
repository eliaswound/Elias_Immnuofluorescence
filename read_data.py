import os
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = 'Path to base'
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

np.save("healthy", heal_avg)
np.save("sick", sic_avg)
#print("sick", sic_avg, "healthy", heal_avg)

'''
fig1 = plt.figure()
ax11 = fig1.add_subplot(121)
ax11.imshow(thresh_heal, cmap='gray')
ax12 = fig1.add_subplot(122)
ax12.imshow(thresh_sic, cmap='gray')
fig1.suptitle("Threshold Masks")
plt.show()
'''
