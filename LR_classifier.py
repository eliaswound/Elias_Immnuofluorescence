import numpy as np
import os.path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


BASE_DIR = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/immunofluorescence_PCD/'

wt = np.load(os.path.join(BASE_DIR, "wt.npy"))
pcd35 = np.load(os.path.join(BASE_DIR, "pcd35.npy"))
pcd36 = np.load(os.path.join(BASE_DIR, "pcd36.npy"))
pcd38 = np.load(os.path.join(BASE_DIR, "pcd38.npy"))

#bootstrapping, 100 rounds
# keep track of number of classified positives in each round
predict_wt = np.zeros(100)
predict_35 = np.zeros(100)
predict_36 = np.zeros(100)
predict_38 = np.zeros(100)


for i in range(100):

	# pick 200 random wildtypes for training
	np.random.shuffle(wt)
	train_healthy = wt[:200]
	test_wt = wt[200:]
	np.random.shuffle(pcd36)
	train_sick = pcd36[:200]
	test_36 = pcd36[200:]



	train = np.concatenate((train_healthy, train_sick))

	train_labels = np.concatenate((np.zeros(train_healthy.shape[0]), np.ones(train_sick.shape[0])))


	svm = SVC()
	svm.fit(train.reshape(-1,1), train_labels.ravel())

	#train_pred = svm.predict(train.reshape(-1,1))
	predict_wt[i] = svm.predict(test_wt.reshape(-1,1)).mean()
	predict_35[i] = svm.predict(pcd35.reshape(-1,1)).mean()
	predict_36[i] = svm.predict(test_36.reshape(-1,1)).mean()
	predict_38[i] = svm.predict(pcd38.reshape(-1,1)).mean()

#confuse = confusion_matrix(train_labels.ravel(), train_pred, labels=[0, 1])
#print("train confuse", confuse)
print("wt ",predict_wt.mean(), np.percentile(predict_wt, 5), np.percentile(predict_wt, 95))
print("35 ",predict_35.mean(), np.percentile(predict_35, 5), np.percentile(predict_35, 95))
print("36 ",predict_36.mean() ,np.percentile(predict_36, 5), np.percentile(predict_36, 95))
print("38 ",predict_38.mean(), np.percentile(predict_38, 5), np.percentile(predict_38, 95))

