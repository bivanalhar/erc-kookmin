import numpy as np
import csv

from sklearn.metrics import roc_auc_score

"""
This page is special to calculate the AUC Score of the
chosen prediction (in this case, RNN and CNN)
"""

def one_hot(label):
	return [float(label == '0'), float(label == '1'), float(label == '2'), float(label == '3'), float(label == '5')]

def extract_value(file):
	f = open(file, 'r')

	value_temp = []
	final_value = []
	for line in f:
		# print(len(line))
		assert(len(line) in [22, 62, 82])

		if len(line) == 22:
			for index in [2, 6, 10, 14, 18]:
				value_temp.append(float(line[index : index+2]))
		elif len(line) == 62:
			for index in [2, 14, 26, 38, 50]:
				value_temp.append(float(line[index : index+10]))
		elif len(line) == 82:
			for index in [2, 18, 34, 50, 66]:
				value_temp.append(float(line[index : index+14]))

		final_value.append(value_temp)
		value_temp = []

	# print(final_value)
	return final_value

def extract_one_hot(file):
	f = open(file, 'r')

	final_one_hot = []
	for line in f:
		line = line.strip()
		final_one_hot.append(one_hot(line))

	return final_one_hot

def compute_auc(final_value, final_one_hot):
	auc_score = [None for i in range(5)]

	#first, compute the AUC for each respective label
	for label in range(5):
		value_temp = [final_value[i][label] for i in range(len(final_value))]
		label_temp = [final_one_hot[i][label] for i in range(len(final_one_hot)) if (i % 5 == 4 or i == len(final_one_hot) - 1)]
		# print(value_temp)

		auc_score[label] = roc_auc_score(label_temp, value_temp)

	return sum(auc_score)/float(len(auc_score))

final_value = extract_value("170612_erc_cnn_pred.txt")
final_one_hot = extract_one_hot("ground_truth/a5p4_stage.txt")

#now calculating the AUC of the prediction
print(compute_auc(final_value, final_one_hot))                                                                                                                                                                                                                                                                                                         