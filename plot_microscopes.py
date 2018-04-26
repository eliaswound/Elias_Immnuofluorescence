import numpy as np
import os.path
import matplotlib.pyplot as plt


BASE_DIR = '/home/yasaman/Mennella_Lab/immunofluorescence_PCD/'

spin_healthy = np.load(os.path.join(BASE_DIR, "spin_healthy.npy"))
wide_healthy = np.load(os.path.join(BASE_DIR, "wide_healthy.npy"))
spin_sick = np.load(os.path.join(BASE_DIR, "spin_sick.npy"))
wide_sick = np.load(os.path.join(BASE_DIR, "wide_sick.npy"))

spin = np.concatenate((spin_healthy, spin_sick))
wide = np.concatenate((wide_healthy, wide_sick))



spin_labels = np.concatenate((np.zeros(spin_healthy.shape[0]), np.ones(spin_sick.shape[0])))
wide_labels = np.concatenate((np.zeros(wide_healthy.shape[0]), np.ones(wide_sick.shape[0])))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist((spin[spin_labels == 0], spin[spin_labels==1]), 20, normed=True, histtype='step', label=("spinning disc healthy", "spinning disc pcd"), linestyle='dotted')

plt.legend()
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.hist((wide[wide_labels==0], wide[wide_labels==1]), 20, normed=True, histtype='step', label=("widefield healthy", "widefield pcd"), linestyle='solid')

plt.legend()
plt.show()



#
