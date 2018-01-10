import numpy as np
import os.path
from sklearn.svm import SVC
import matplotlib.pyplot as plt


BASE_DIR = '/home/yasaman/Documents/Winter18/fluorescence/immunofluorescence_PCD/'

test_healthy = np.load(os.path.join(BASE_DIR, "test_healthy.npy"))
train_healthy = np.load(os.path.join(BASE_DIR, "train_healthy.npy"))
test_sick = np.load(os.path.join(BASE_DIR, "test_sick.npy"))
train_sick = np.load(os.path.join(BASE_DIR, "train_sick.npy"))

healthy = np.concatenate((train_healthy, test_healthy))
sick = np.concatenate((train_sick, test_sick))
all_scores = np.concatenate((healthy, sick))
all_labels = np.concatenate((np.zeros(healthy.shape[0]), np.ones(sick.shape[0])))
all_scores = all_scores.reshape(-1,1)
all_labels = all_labels.reshape(-1, 1)
all_data = np.concatenate((all_scores, all_labels), axis=1)

test_ratio = 0.2
test_num = int((test_ratio * all_data.shape[0])//1)
num_runs = 10
error = np.zeros(num_runs)

for i in range(10):
    np.random.shuffle(all_data)
    test_data = all_data[:test_num]
    train_data = all_data[test_num:]
    svm = SVC()
    svm.fit(train_data[:,0].reshape(-1, 1), train_data[:,1].ravel())
    score = svm.score(test_data[:,0].reshape(-1,1), test_data[:,1].ravel())
    error[i] = score


fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(np.arange(num_runs), error, 'ro')
ax.plot(1, error.mean(), 'k+')
ax.plot(1, np.median(error), 'bs')
ax.legend(["error", "mean", "median"])
fig1.suptitle("Error for different partitionings of dataset")

fig1.savefig("error.png")

print(error)






