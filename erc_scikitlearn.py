from sklearn import svm, metrics, tree, ensemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# import tensorflow as tf
import numpy as np
import csv

"""
This code is to implement the ERC Kookmin Project, in which
we analyze the sleeping pattern of the selected patients (in this case, 4)
2 of them will be made as the training set, one will be the validation set
and another one will be made for the testing test
"""

#this function is purposed to pair up the data as well as the label
#please be noted that the data file is in csv format while the 
def data_label_preprocess(data_file, label_file):

	data_list = []
	label_list = []

	#first phase : storing all the data into the appropriate data structure
	with open(data_file) as data_csv:
		csvread = csv.reader(data_csv)
		for row in csvread:
			data_list.append([np.float32(i) for i in row])
	data_csv.close()

	#second phase : storing all the label into the appropriate data structure
	label_f = open(label_file, 'r')
	for line in label_f:
		label_list.append(line.strip())
	label_f.close()

	# print(label_list)
	return data_list, label_list

#the function to compute the accuracy of the classifier towards the dataset
def compute_accuracy(classifier, data_set, label_set):
	result = classifier.predict(data_set)
	decision = [int(result[i] == label_set[i]) for i in range(len(label_set))]

	# print([i for i in result])
	return (sum(decision) / float(len(decision)))

def compute_auc(classifier, data_train, label_train, data_set, label_set):
	"""
	Computing the AUC Score for the multiclass classifier
	Here we are computing the AUC Score for each label, then we averaged
	them all together (using the micro approach)
	"""
	# result_prob = classifier.predict_proba(data_set)
	label_list = ['0', '1', '2', '3', '5']
	
	auc_list = [None for i in range(len(label_list))]
	#Computing the AUC Score for each label
	for label in label_list:
		idx = label_list.index(label)
		# result_prob_temp = [result_prob[i][idx] for i in range(len(result_prob))]
		label_set_temp = [int(label == label_set[i]) for i in range(len(label_set))]
		label_train_temp = [int(label == label_train[i]) for i in range(len(label_train))]
		
		classifier.fit(data_train, label_train_temp)
		result_prob_temp = [classifier.predict_proba(data_set)[i][1] for i in range(len(data_set))]

		auc_list[idx] = metrics.roc_auc_score(label_set_temp, result_prob_temp)

	return sum(auc_list) / float(len(auc_list))

data_list_1, label_list_1 = data_label_preprocess("feature_extraction/finaldb_a5p1_6ch.csv", "ground_truth/a5p1_stage.txt")
data_list_2, label_list_2 = data_label_preprocess("feature_extraction/finaldb_a5p2_6ch.csv", "ground_truth/a5p2_stage.txt")
data_list_3, label_list_3 = data_label_preprocess("feature_extraction/finaldb_a5p3_6ch.csv", "ground_truth/a5p3_stage.txt")
data_list_4, label_list_4 = data_label_preprocess("feature_extraction/finaldb_a5p4_6ch.csv", "ground_truth/a5p4_stage.txt")

#forming up the train, val and test dataset
data_train, label_train = data_list_1 + data_list_2, label_list_1 + label_list_2
data_val, label_val = data_list_3, label_list_3
data_test, label_test = data_list_4, label_list_4

#begin the Scikit-Learn module implementation

#First classifier : Nu-SVC
nu_list = [0.005, 0.01, 0.05, 0.1]

best_nu = 0.0
best_val_acc = 0.0
test_acc = None

for nu in nu_list:
	classifier_nusvm = svm.NuSVC(nu = nu)
	classifier_nusvm.fit(data_train, label_train)

	val_accuracy = compute_accuracy(classifier_nusvm, data_val, label_val)

	if val_accuracy > best_val_acc:
		best_val_acc = val_accuracy
		best_nu = nu
		test_acc = compute_accuracy(classifier_nusvm, data_test, label_test)

print("Result using Nu-SVC classifier")
print("Best Validation Accuracy", best_val_acc*100, "percent\nBest value for Nu", best_nu,"\nTest accuracy for the best setting is", test_acc * 100, "percent\n")

#Second Classifier : Multilayer Perceptron
alpha_list = [0.0001, 0.001, 0.01]
learning_rate_list = [0.001, 0.01, 0.1, 1]

best_alpha = 0.0
best_learn_rate = 0.0
best_val_acc = 0.0
test_acc = None
auc_score = None

for alpha in alpha_list:
	for learning_rate in learning_rate_list:
		classifier_mlp = MLPClassifier(alpha = alpha, learning_rate_init = learning_rate)
		classifier_mlp.fit(data_train, label_train)

		val_accuracy = compute_accuracy(classifier_mlp, data_val, label_val)

		if val_accuracy > best_val_acc:
			best_val_acc = val_accuracy
			best_alpha = alpha
			best_learn_rate = learning_rate

			predict_prob = classifier_mlp.predict_proba(data_test)
			test_acc = compute_accuracy(classifier_mlp, data_test, label_test)
			auc_score = compute_auc(classifier_mlp, data_train, label_train, data_test, label_test)

print("Result using MLP Classifier")
print("Best Validation Accuracy", best_val_acc*100, "percent\nBest value for alpha", best_alpha,"\nBest value for learning rate", best_learn_rate, "\nTest accuracy for the best setting is", test_acc * 100, "percent\nAUC Score of the best setting", auc_score, "\n")

#Third Classifier : SVC
C_list = [0.001, 0.01, 0.1, 1]

best_C = 0.0
best_val_acc = 0.0
test_acc = None

for C in C_list:
	classifier_svm = svm.SVC(C = C)
	classifier_svm.fit(data_train, label_train)

	val_accuracy = compute_accuracy(classifier_svm, data_val, label_val)

	if val_accuracy > best_val_acc:
		best_val_acc = val_accuracy
		best_C = C
		test_acc = compute_accuracy(classifier_svm, data_test, label_test)

print("Result using SVC classifier")
print("Best Validation Accuracy", best_val_acc*100, "percent\nBest value for C", best_C,"\nTest accuracy for the best setting is", test_acc * 100, "percent\n")

#Fourth Classifier : Logistic Regression (l1 Penalty)
C_list = [0.001, 0.01, 0.1, 1]

best_C = 0.0
best_val_acc = 0.0
test_acc = None

for C in C_list:
	classifier_logl1 = LogisticRegression(C = C, penalty = 'l1')
	classifier_logl1.fit(data_train, label_train)

	val_accuracy = compute_accuracy(classifier_logl1, data_val, label_val)

	if val_accuracy > best_val_acc:
		best_val_acc = val_accuracy
		best_C = C
		test_acc = compute_accuracy(classifier_logl1, data_test, label_test)
		auc_score = compute_auc(classifier_logl1, data_train, label_train, data_test, label_test)

print("Result using Logistic Regression (l1 penalty)")
print("Best Validation Accuracy", best_val_acc*100, "percent\nBest value for C", best_C,"\nTest accuracy for the best setting is", test_acc * 100, "percent\nAUC Score of the best setting", auc_score, "\n")

#Fifth Classifier : Logistic Regression (l2 Penalty)
C_list = [0.001, 0.01, 0.1, 1]

best_C = 0.0
best_val_acc = 0.0
test_acc = None

for C in C_list:
	classifier_logl2 = LogisticRegression(C = C)
	classifier_logl2.fit(data_train, label_train)

	val_accuracy = compute_accuracy(classifier_logl2, data_val, label_val)

	if val_accuracy > best_val_acc:
		best_val_acc = val_accuracy
		best_C = C
		test_acc = compute_accuracy(classifier_logl2, data_test, label_test)
		auc_score = compute_auc(classifier_logl2, data_train, label_train, data_test, label_test)

print("Result using Logistic Regression (l2 penalty)")
print("Best Validation Accuracy", best_val_acc*100, "percent\nBest value for C", best_C,"\nTest accuracy for the best setting is", test_acc * 100, "percent\nAUC Score of the best setting", auc_score, "\n")

#Sixth Classifier : Decision Tree Classifier
classifier_tree = tree.DecisionTreeClassifier()
classifier_tree.fit(data_train, label_train)

test_acc = compute_accuracy(classifier_tree, data_test, label_test)
auc_score = compute_auc(classifier_tree, data_train, label_train, data_test, label_test)

print("Result using Decision Tree Classifier")
print("Test accuracy is", test_acc * 100, "percent\nAUC Score is", auc_score, "\n")

#Seventh Classifier : Random Forest Classifier
classifier_forest = ensemble.RandomForestClassifier()
classifier_forest.fit(data_train, label_train)

test_acc = compute_accuracy(classifier_forest, data_test, label_test)
auc_score = compute_auc(classifier_forest, data_train, label_train, data_test, label_test)

print("Result using Random Forest Classifier")
print("Test accuracy is", test_acc * 100, "percent\nAUC Score is", auc_score, "\n")

#Eighth Classifier : Bagging Classifier
classifier_bagging = ensemble.BaggingClassifier()
classifier_bagging.fit(data_train, label_train)

test_acc = compute_accuracy(classifier_bagging, data_test, label_test)
auc_score = compute_auc(classifier_bagging, data_train, label_train, data_test, label_test)

print("Result using Bagging Classifier")
print("Test accuracy is", test_acc * 100, "percent\nAUC Score is", auc_score, "\n")

#Final Classifier : Voting Classifier
classifier_voting = CalibratedClassifierCV(ensemble.VotingClassifier(estimators = 
	[('mlp', classifier_mlp), ('logl1', classifier_logl1), ('logl2', classifier_logl2), 
	('tree', classifier_tree), ('forest', classifier_forest), ('bagging', classifier_bagging)],
	voting = 'soft'))
classifier_voting.fit(data_train, label_train)

test_acc = compute_accuracy(classifier_voting, data_test, label_test)
auc_score = compute_auc(classifier_voting, data_train, label_train, data_test, label_test)

print("Result using Voting Classifier")
print("Test accuracy is", test_acc * 100, "percent\nAUC Score is", auc_score, "\n")