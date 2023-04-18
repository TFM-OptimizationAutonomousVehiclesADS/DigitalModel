import numpy as np
from PIL import Image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
# import keras.backend as K
import base64

tp = 3.0
tn = 5.0
fp = 1.0
fn = 2.0

precision = (tp) / (tp + fp)
print(precision)


def precision_threshold(threshold=0.5):
    def threshold_fn(y_true, y_pred):
        y_pred = K.cast(K.greater_equal(y_pred, threshold), 'float32')
        print(y_pred)
        return y_true, y_pred

    def precision(y_true, y_pred):
        y_true, y_pred = threshold_fn(y_true, y_pred)
        true_positives = K.sum(y_true * y_pred)
        print(true_positives)
        false_positives = K.sum((1 - y_true) * y_pred)
        print(false_positives)
        precision = true_positives / (true_positives + false_positives )
        return precision

    return precision

def accuracy_threshold(threshold=0.5):
    def threshold_fn(y_true, y_pred):
        y_pred = K.cast(K.greater_equal(y_pred, threshold), 'float32')
        return y_true, y_pred

    def accuracy(y_true, y_pred):
        y_true, y_pred = threshold_fn(y_true, y_pred)
        true_positives = K.sum(y_true * y_pred)
        false_positives = K.sum((1 - y_true) * y_pred)
        false_negatives = K.sum(y_true * (1 - y_pred))
        true_negatives = K.sum((1 - y_true) * (1 - y_pred))
        correct_predictions = true_positives + true_negatives
        total_predictions = true_positives + true_negatives + false_negatives + false_positives
        accuracy = correct_predictions / (total_predictions + K.epsilon())
        return accuracy
    return accuracy

y_true = tf.convert_to_tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=tf.float32)
y_pred = tf.convert_to_tensor([0.6, 0.7, 0.55, 0, 0, 0, 0, 0, 0.5, 0.4, 0.4], dtype=tf.float32)
print(precision_threshold()(y_true, y_pred))
print(accuracy_threshold()(y_true, y_pred))
