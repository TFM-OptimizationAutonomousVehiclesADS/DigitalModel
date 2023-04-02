from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def true_positives(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.int32)  # se binariza la predicción
    y_true_binary = tf.cast(y_true, dtype=tf.int32)  # se binariza la etiqueta verdadera
    true_positives = tf.reduce_sum(y_true_binary * y_pred_binary)  # se calculan los verdaderos positivos
    return true_positives

def false_positives(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.int32)  # se binariza la predicción
    y_true_binary = tf.cast(y_true, dtype=tf.int32)  # se binariza la etiqueta verdadera
    false_positives = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)  # se calculan los falsos positivos
    return false_positives

def true_negatives(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.int32)  # se binariza la predicción
    y_true_binary = tf.cast(y_true, dtype=tf.int32)  # se binariza la etiqueta verdadera
    true_negatives = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary))  # se calculan los verdaderos negativos
    return true_negatives

def false_negatives(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.int32)  # se binariza la predicción
    y_true_binary = tf.cast(y_true, dtype=tf.int32)  # se binariza la etiqueta verdadera
    false_negatives = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))  # se calculan los falsos negativos
    return false_negatives


y_pred = np.array([0.23, 0.24, 0.55, 1.0, 0.0, 0.25, 0.8, 0.6])
y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])

tp = false_positives(y_true, y_pred, 0.5)
print(tp)
