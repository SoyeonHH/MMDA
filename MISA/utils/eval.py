import codecs
from sklearn import metrics
import numpy as np
import os

def dense(y):
	label_y = []
	for i in range(len(y)):
		for j in range(len(y[i])):
			label_y.append(y[i][j])

	return label_y

def get_accuracy(y, y_pre):
	samples = len(y)
	count = 0.0
	for i in range(samples):
		y_true = 0
		all_y = 0
		for j in range(len(y[i])):
			if y[i][j] > 0 and y_pre[i][j] > 0:
				y_true += 1
			if y[i][j] > 0 or y_pre[i][j] > 0:
				all_y += 1
		if all_y <= 0:
			all_y = 1

		count += float(y_true) / float(all_y)
	acc = float(count) / float(samples)
	acc = round(acc, 4)
	return acc

def get_metrics(y, y_pre, average='macro'):
	"""
	:param y:1871*6
	:param y_pre: 1871*6
	:return: acc, macro_f1, macro_precision, macro_recall
	"""
	# y = y.cpu().detach().numpy()
	# y_pre = y_pre.cpu().detach().numpy()
	test_labels = dense(y)
	test_pred = dense(y_pre)

	test_labels = y
	test_pred = y_pre

	acc = get_accuracy(test_labels, test_pred)

	if average == 'macro':
		# macro
		f1 = metrics.f1_score(test_labels, test_pred, average='macro')
		precision = metrics.precision_score(test_labels, test_pred, average='macro')
		recall = metrics.recall_score(test_labels, test_pred, average='macro')

	elif average == 'micro':
		# micro
		f1 = metrics.f1_score(test_labels, test_pred, average='micro')
		precision = metrics.precision_score(test_labels, test_pred, average='micro')
		recall = metrics.recall_score(test_labels, test_pred, average='micro')

	elif average == 'weighted':
		# weighted
		f1 = metrics.f1_score(test_labels, test_pred, average='weighted')
		precision = metrics.precision_score(test_labels, test_pred, average='weighted')
		recall = metrics.recall_score(test_labels, test_pred, average='weighted')

	return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}
	
