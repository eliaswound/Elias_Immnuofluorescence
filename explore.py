import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
from PIL import Image
from read_data import process_image

# Data PATH
BASE_DIR = '/home/eguan'
HEALTH_DIR ='healthy/'
SICK_DIR = 'sick/'

healthy = os.listdir(os.path.join(BASE_DIR, HEALTH_DIR))
sick = os.listdir(os.path.join(BASE_DIR, SICK_DIR))

healthy.sort()
sick.sort()

picked_healthy = 3
picked_sick = 3

healthy_im = Image.open(os.path.join(BASE_DIR, HEALTH_DIR, healthy[picked_healthy]))
sick_im = Image.open(os.path.join(BASE_DIR, SICK_DIR, sick[picked_sick]))

heal_arr = np.asarray(healthy_im, dtype='float32')
sic_arr = np.asarray(sick_im, dtype='float32')

heal_corr= process_image(healthy_im)
sic_corr = process_image(sick_im)



fig1 = plt.figure()
ax11 = fig1.add_subplot(121)
ax11.imshow(heal_arr[:,:,0], cmap='gray')
ax11.set_title("Red channel")
ax12 = fig1.add_subplot(122)
ax12.imshow(heal_arr[:,:,1], cmap='gray')
ax12.set_title("Green Channel")
fig1.suptitle("Healthy, "+ healthy[picked_healthy]+ ", Corr is" + str(heal_corr))

fig2 = plt.figure()
ax21 = fig2.add_subplot(121)
ax21.imshow(sic_arr[:,:,0], cmap='gray')
ax21.set_title("Red channel")
ax22 = fig2.add_subplot(122)
ax22.imshow(sic_arr[:,:,1], cmap='gray')
ax22.set_title("Green Channel")
fig2.suptitle("Sick, Corr is "+ sick[picked_sick]+ "Corr is" + str(sic_corr))
plt.show()
