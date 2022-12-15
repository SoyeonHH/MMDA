import codecs
from sklearn import metrics
import numpy as np
import os

def get_accuracy(y, y_pre):
	return metrics.accuracy_score(y, y_pre)

def get_metrics(y, y_pre):
	"""
	:param y:1871*6
	:param y_pre: 1871*6
	:return: 
	"""

	macro_f1 = metrics.f1_score(y, y_pre, average='macro')
	macro_precision = metrics.precision_score(y, y_pre, average='macro')
	macro_recall = metrics.recall_score(y, y_pre, average='macro')
	acc = get_accuracy(y, y_pre)
	return acc, macro_f1, macro_precision, macro_recall
