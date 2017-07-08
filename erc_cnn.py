import tensorflow as tf
import numpy as np
import csv
# import util

import matplotlib.pyplot as plt

"""
Method to implement : Recurrent Neural Network (RNN)
More specifically, we will implement the RNN-LSTM method
towards the data of sleeping pattern prediction
"""
def one_hot(label):
	return [float(label == '0'), float(label == '1'), float(label == '2'), float(label == '3'), float(label == '5')]

def seq2point_preprocess(data_file, label_file):
	#here we need to note that the data_file is always in csv format
	#while the label_file is always in txt format

	padding_zero = [0 for i in range(25)]

	data_list = []
	label_list = []

	#first phase : storing all the data into the appropriate data structure
	with open(data_file) as data_csv:
		csvread = csv.reader(data_csv)
		for row in csvread:
			data_list.append([np.float32(i) for i in row])
	data_csv.close()

	#getting the z-score for the data list
	data_list = np.transpose(np.asarray(data_list))
	mean_list = np.mean(data_list, axis = 0)
	stdev_list = np.std(data_list, axis = 0)

	# data_list = np.transpose(data_list)

	data_list = (data_list - mean_list) / stdev_list
	data_list = np.clip(np.transpose(data_list), -4, 4)
	# print(data_list)

	# util.RaiseNotDefined()

	#second phase : storing all the label into the appropriate data structure
	label_f = open(label_file, 'r')
	for line in label_f:
		label_list.append(line.strip())
	label_f.close()

	#third phase : divide the data and label into group of 5
	data_temp = []
	label_temp = []

	data_list_final = []
	label_list_final = []

	for i in range(len(data_list) - 5):
		# print(i)
		for j in range(6):
			data_temp.append(data_list[i + j])
			label_temp.append(label_list[i + j])
		data_list_final.append(data_temp)
		label_list_final.append(one_hot(label_temp[-1]))
		data_temp = []
		label_temp = []

	for i in range(len(data_list_final)):
		for j in range(4):
			data_list_final[i].insert(0, padding_zero)

	# print(label_list)
	return data_list_final, label_list_final

data_1, label_1 = seq2point_preprocess("feature_extraction/finaldb_a5p1_6ch.csv", "ground_truth/a5p1_stage.txt")
data_2, label_2 = seq2point_preprocess("feature_extraction/finaldb_a5p2_6ch.csv", "ground_truth/a5p2_stage.txt")
data_3, label_3 = seq2point_preprocess("feature_extraction/finaldb_a5p3_6ch.csv", "ground_truth/a5p3_stage.txt")
data_4, label_4 = seq2point_preprocess("feature_extraction/finaldb_a5p4_6ch.csv", "ground_truth/a5p4_stage.txt")

# print(label_1)

#defining the data being used for the training, validation and testing
train_data, train_label = data_1 + data_2, label_1 + label_2
val_data, val_label = data_3, label_3
test_data, test_label = data_4, label_4

####################################################################
############# BEGIN : Implementing the CNN Model ###################
####################################################################

#defining the hyperparameter
training_epoch = 1000
hidden_nodes = 128
batch_size = 128
learning_rate = 0.0001
dropout_rate = 0.2
l2_regularize = True
reg_param = 0.1
kernel_size = 3

#now defining the model for the RNN-LSTM

data = tf.placeholder(tf.float32, [None, 10, 25])
target = tf.placeholder(tf.float32, [None, 5])

with tf.device("/gpu:0"):
	weight_1 = tf.Variable(tf.random_normal(shape = [kernel_size, int(data.get_shape()[2]), 32]))
	bias_1 = tf.Variable(tf.constant(0.1, shape = [32]))

	weight_2 = tf.Variable(tf.random_normal(shape = [kernel_size, 32, 64]))
	bias_2 = tf.Variable(tf.constant(0.1, shape = [64]))

	conv1 = tf.nn.conv1d(data, weight_1, stride = 1, padding = 'VALID')
	# print(conv1.get_shape().as_list())
	conv1 = tf.nn.relu(conv1 + bias_1)
	# print(conv1.get_shape())

	conv2 = tf.nn.conv1d(conv1, weight_2, stride = 1, padding = 'VALID')
	conv2 = tf.nn.relu(conv2 + bias_2)
	# print(conv2.get_shape())

	conv2_flat = tf.reshape(conv2, [-1, 6 * 64])
	dense = tf.layers.dense(inputs=conv2_flat, units=128, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate)

	logits = tf.layers.dense(inputs = dropout, units = 5)
	logits_softmax = tf.nn.softmax(logits)

	loss = tf.losses.softmax_cross_entropy(onehot_labels = target, logits = logits) + reg_param*(tf.nn.l2_loss(weight_1)+tf.nn.l2_loss(bias_1)+tf.nn.l2_loss(weight_2)+tf.nn.l2_loss(bias_2))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

	correct = tf.equal(tf.argmax(target, 1), tf.argmax(logits, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	#initializing all the trainable parameters here
	init_op = tf.global_variables_initializer()

f = open("170706_erc_cnn_stride6_zscore.txt", 'w')
f.write("Result of the experiment\n\n")

batch_size_list = [128]
hidden_layer_list = [128]
learning_rate_list = [1e-3]
epoch_list_run = [2000]
dropout_list = [0.9, 0.8, 0.7, 0.5, 0.3]
regularizer_parameter = [0.0001, 0.001, 0.01, 0.1]
l2Regularize_list = [True]

count_exp = 1

for batch_size1 in batch_size_list:
	for training_epoch1 in epoch_list_run:
		for learning_rate1 in learning_rate_list:
			for hidden_node1 in hidden_layer_list:
				for dropout_rate1 in dropout_list:
					for l2Reg in l2Regularize_list:
						for reg_param1 in regularizer_parameter:

							batch_size = batch_size1
							hidden_nodes = hidden_node1
							learning_rate = learning_rate1
							training_epoch = training_epoch1
							dropout_rate = dropout_rate1
							l2_regularize = l2Reg
							reg_param = reg_param1
							epoch_list = []
							cost_list = []	

							print("batch size = " + str(batch_size))
							print("hidden nodes = " + str(hidden_nodes))
							print("learning rate = " + str(learning_rate))
							print("training epoch = " + str(training_epoch))
							print("dropout rate = " + str(1 - dropout_rate))
							print("l2Reg = " + str(l2_regularize))
							print("reg_param = " + str(reg_param))

							f.write("setting up the experiment with\n")
							f.write("batch size = " + str(batch_size) + ", hidden nodes = " + str(hidden_nodes) + ", learning rate = " + str(learning_rate) + "\n")
							f.write("training epoch = " + str(training_epoch) + ", dropout rate = " + str(1 - dropout_rate) + ", reg_param = " + str(reg_param) + "\n\n")

							with tf.Session() as sess:
								sess.run(init_op)

								for epoch in range(training_epoch):
									epoch_list.append(epoch + 1)
									ptr = 0
									avg_cost = 0.
									no_of_batches = int(len(train_data) / batch_size)
									# no_of_batches = 1

									for i in range(no_of_batches):
										batch_in, batch_out = train_data[ptr:ptr+batch_size], train_label[ptr:ptr+batch_size]
										ptr += batch_size
										# target_ = sess.run([target], feed_dict = {data : batch_in, target : batch_out})

										_, cost_ = sess.run([optimizer, loss], feed_dict = {data : batch_in, target : batch_out})

										avg_cost += cost_ / no_of_batches
									# print("loss function = " + str(avg_cost))
									cost_list.append(avg_cost)

									# sess.run(target_exp, feed_dict = {data : train_data, target : train_label})
									# sess.run(arg_pred, feed_dict = {data : train_data, target : train_label})

									if epoch in [99, 199, 299, 499, 699, 999, 1099, 1199, 1299, 1499, 1699, 1999]:
										f.write("During the " + str(epoch+1) + "-th epoch:\n")
										f.write("Training Accuracy = " + str(sess.run(accuracy, feed_dict = {data : train_data, target : train_label})) + "\n")
										f.write("Validation Accuracy = " + str(sess.run(accuracy, feed_dict = {data : val_data, target : val_label})) + "\n")
										f.write("Testing Accuracy = " + str(sess.run(accuracy, feed_dict = {data : test_data, target : test_label})) + "\n\n")
								print("Optimization Finished")

								# # saver.save(sess, save_path)
								# for i in range(len(test_data)):
								# 	pred = sess.run(logits_softmax, feed_dict = {data : test_data[i:i+1]})
								# 	f.write(str(pred[0]) + "\n")

								plt.plot(epoch_list, cost_list)
								plt.xlabel("Epoch (dropout = " + str(dropout_rate) + ";l2Reg = " + str(reg_param) + ";learn_rate = " + str(learning_rate) + ")")
								plt.ylabel("Cost Function")

								training_accuracy = sess.run(accuracy, feed_dict = {data : train_data, target : train_label})
								validation_accuracy = sess.run(accuracy, feed_dict = {data : val_data, target : val_label})
								testing_accuracy = sess.run(accuracy, feed_dict = {data : test_data, target : test_label})
								
								print("Finished Accuracy Calculation. Now saving the learning curve")

								plt.title("Train Acc = " + str(training_accuracy * 100) + "\nTest Acc = " + str(testing_accuracy * 100))

								plt.savefig("170706_fig_cnn_stride6_zscore Exp " + str(count_exp) + ".png")

								plt.clf()

								count_exp += 1