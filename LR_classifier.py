import numpy as np
import os.path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE_DIR = '/home/yasaman/Mennella_Lab/immunofluorescence_PCD/'

test_healthy = np.load(os.path.join(BASE_DIR, "test_healthy.npy"))
#train_healthy = np.load(os.path.join(BASE_DIR, "train_healthy.npy"))
test_sick = np.load(os.path.join(BASE_DIR, "test_sick.npy"))
test_names = np.load(os.path.join(BASE_DIR, "test_names_hs.npy"))
#train_sick = np.load(os.path.join(BASE_DIR, "train_sick.npy"))

#train = np.concatenate((train_healthy, train_sick))
test = np.concatenate((test_healthy, test_sick))
#all_scores = np.concatenate((healthy, sick))
#all_labels = np.concatenate((np.zeros(healthy.shape[0]), np.ones(sick.shape[0])))
#all_scores = all_scores.reshape(-1,1)
#all_labels = all_labels.reshape(-1, 1)
#all_data = np.concatenate((all_scores, all_labels), axis=1)

#train_labels = np.concatenate((np.zeros(train_healthy.shape[0]), np.ones(train_sick.shape[0])))
test = test.reshape(-1, 1)
test_labels = np.concatenate((np.zeros(test_healthy.shape[0]), np.ones(test_sick.shape[0])))
test_labels = test_labels.reshape(-1, 1)
test_names = test_names.reshape(-1, 1)
all_test = np.concatenate((test, test_labels, test_names), axis=1)

'''
svm = SVC()
#lr = LogisticRegression()
svm.fit(train.reshape(-1,1), train_labels.ravel())


score = svm.score(test.reshape(-1,1), test_labels.ravel())
predict = svm.predict(test.reshape(-1,1))
confuse = confusion_matrix(test_labels.ravel(), predict, labels=[0, 1])
print(score, confuse)

plt.figure()
plt.plot(train, train_labels, 'ro')
plt.plot(test, test_labels, 'b+')
plt.savefig("data_dist.pdf")
'''
test_ratio = 0.2
test_num = int((test_ratio * all_test.shape[0])//1)
num_runs = 10
error = np.zeros(num_runs)
confusion = np.zeros([num_runs, 2, 2])



for i in range(1):
    np.random.shuffle(all_test)
    test_data = all_test[:test_num]
    train_data = all_test[test_num:]
    svm = SVC()
    svm.fit(train_data[:,0].reshape(-1, 1), train_data[:,1].ravel())
    score = svm.score(test_data[:,0].reshape(-1,1), test_data[:,1].ravel())
    error[i] = score
    predict = svm.predict(test_data[:,0].reshape(-1, 1))
    print(test_data[(test_data[:,1] != predict), 2])
    confusion[i] = confusion_matrix(test_data[:,1], predict, labels=['0.0', '1.0'])
    



fig1 = plt.figure()

ax = fig1.add_subplot(111)
ax.hist(error)
ax.axvline(error.mean(), color='b', linestyle='dashed', linewidth=2)
ax.axvline(np.median(error), color='k', linestyle='dotted', linewidth=2)
ax.legend(["mean", "median"])
fig1.suptitle("Error for different partitionings of dataset")

fig1.savefig("error2.pdf")

avg_confusion = confusion.mean(axis=0)
avg_confusion = avg_confusion/test_num
print(avg_confusion)



