import numpy as np
import os.path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


BASE_DIR = '/hpf/largeprojects/agoldenb/yasaman/Mennella_Lab/immunofluorescence_PCD/'

wt = np.load(os.path.join(BASE_DIR, "wt.npy"))
wt17 = np.load(os.path.join(BASE_DIR, "wt17.npy"))
wt23 = np.load(os.path.join(BASE_DIR, "wt23.npy"))
wt40E = np.load(os.path.join(BASE_DIR, "wt40E.npy"))
wt40Z = np.load(os.path.join(BASE_DIR, "wt40Z.npy"))
pcd17 = np.load(os.path.join(BASE_DIR, "pcd17.npy"))
pcd36 = np.load(os.path.join(BASE_DIR, "pcd36.npy"))
pcd23 = np.load(os.path.join(BASE_DIR, "pcd23.npy"))
pcd40 = np.load(os.path.join(BASE_DIR, "pcd40.npy"))

#bootstrapping, 100 rounds
# keep track of number of classified positives in each round
predict_wt = np.zeros(100)
predict_wt17 = np.zeros(100)
predict_wt23 = np.zeros(100)
predict_wt40E = np.zeros(100)
predict_wt40Z = np.zeros(100)
predict_17 = np.zeros(100)
predict_36 = np.zeros(100)
predict_23 = np.zeros(100)
predict_40 = np.zeros(100)


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
	predict_wt17[i] = svm.predict(wt17.reshape(-1,1)).mean()
	predict_wt23[i] = svm.predict(wt23.reshape(-1,1)).mean()
	predict_wt40E[i] = svm.predict(wt40E.reshape(-1,1)).mean()
	predict_wt40Z[i] = svm.predict(wt40Z.reshape(-1,1)).mean()
	predict_17[i] = svm.predict(pcd17.reshape(-1,1)).mean()
	predict_36[i] = svm.predict(test_36.reshape(-1,1)).mean()
	predict_23[i] = svm.predict(pcd23.reshape(-1,1)).mean()
	predict_40[i] = svm.predict(pcd40.reshape(-1,1)).mean()

#confuse = confusion_matrix(train_labels.ravel(), train_pred, labels=[0, 1])
#print("train confuse", confuse)
print("wt ",predict_wt.mean(), np.percentile(predict_wt, 5), np.percentile(predict_wt, 95))
print("wt 17 ", predict_wt17.mean(), np.percentile(predict_wt17, 5), np.percentile(predict_wt17, 95))
print("wt 23", predict_wt23.mean(), np.percentile(predict_wt23, 5), np.percentile(predict_wt23, 95))
print("wt 40E ", predict_wt40E.mean(), np.percentile(predict_wt40E, 5), np.percentile(predict_wt40E, 95))
print("wt 40Z ", predict_wt40Z.mean(), np.percentile(predict_wt40Z, 5), np.percentile(predict_wt40Z, 95))
print("17 ",predict_17.mean(), np.percentile(predict_17, 5), np.percentile(predict_17, 95))
print("36 ",predict_36.mean() ,np.percentile(predict_36, 5), np.percentile(predict_36, 95))
print("23 ",predict_23.mean(), np.percentile(predict_23, 5), np.percentile(predict_23, 95))
print("23 ",predict_40.mean(), np.percentile(predict_40, 5), np.percentile(predict_40, 95))

