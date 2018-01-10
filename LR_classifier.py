import numpy as np
import os.path
from sklearn.svm import SVC
import matplotlib.pyplot as plt


BASE_DIR = '/home/yasaman/Documents/Winter18/fluorescence/immunofluorescence_PCD/'

test_healthy = np.load(os.path.join(BASE_DIR, "test_healthy.npy"))
train_healthy = np.load(os.path.join(BASE_DIR, "train_healthy.npy"))
test_sick = np.load(os.path.join(BASE_DIR, "test_sick.npy"))
train_sick = np.load(os.path.join(BASE_DIR, "train_sick.npy"))

train_data = np.concatenate((train_healthy, train_sick))
train_label = np.concatenate((np.zeros(train_healthy.shape[0]), np.ones(train_sick.shape[0])))
#plt.figure()
#plt.plot(train_sick[:,0], train_sick[:,1], 'ro')
#plt.plot(train_healthy[:,0], train_healthy[:,1], 'k+')
#plt.show()
train_data = train_data.reshape(-1, 1)

test_data = np.concatenate((test_healthy, test_sick))
test_label = np.concatenate((np.zeros(test_healthy.shape[0]), np.ones(test_sick.shape[0])))
test_data = test_data.reshape(-1, 1)

#print("train_data", train_data, "train labels", train_label) 

svm = SVC()
svm.fit(train_data, train_label.ravel())

score = svm.score(test_data, test_label.ravel())


print(score)






