import numpy as np
from PIL import Image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
# import keras.backend as K
import base64


def resize_image(im, size_image=None):
    if not size_image:
        size_image = [320, 180]
    im = im.resize((size_image[1], size_image[0]))
    # im = transform(im)
    return im


def get_float_channel_camera(channel_camera):
    dict_to_float = {
        "CAM_FRONT": 1,
        "CAM_BACK": -1
    }
    return dict_to_float.get(channel_camera, -1)


def get_data_and_labels_from_dataset(dataset, size_image=[99, 99]):
    # Get X, y
    X = []
    X_images = []
    X_features = []
    y = []
    for index, row in dataset.iterrows():
        im1 = np.array(resize_image(Image.open(row["filename_resized_image"]), size_image=size_image)) / 255.0
        im2 = np.array(resize_image(Image.open(row["filename_objects_image"]), size_image=size_image)) / 255.0
        im3 = np.array(resize_image(Image.open(row["filename_surfaces_image"]), size_image=size_image)) / 255.0
        ims_array = [im1, im2, im3]

        camera = get_float_channel_camera(row["channel_camera"])
        speed = row["speed"]
        rotation = row["rotation_rate_z"]
        features_array = [camera, speed, rotation]

        X_images.append(ims_array)
        X_features.append(features_array)
        X.append([im1, im2, im3, features_array])
        y.append([row["anomaly"]])

    # X = np.array(X)
    # y = np.array(y)
    return X, y


def convert_X_data_to_images_features_json(X):
    X_full_images = np.array(list(list(zip(*X))[0]))
    X_objects_images = np.array(list(list(zip(*X))[1]))
    X_surfaces_images = np.array(list(list(zip(*X))[2]))
    X_features = np.array(list(list(zip(*X))[3]))
    X_json = {"full_images": X_full_images, "objects_images": X_objects_images, "surfaces_images": X_surfaces_images,
              "features": X_features}
    return X_json


def get_train_test_split(dataset, size_image, size_split, test_size, random_state):
    dataset = dataset.sample(frac=1)
    if size_split:
        dataset = dataset.head(size_split)
    X, y = get_data_and_labels_from_dataset(dataset, size_image)
    X_train, X_tests, y_train, y_tests = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train_json = convert_X_data_to_images_features_json(X_train)
    X_tests_json = convert_X_data_to_images_features_json(X_tests)
    y_train = np.array(y_train)
    y_tests = np.array(y_tests)
    return X_train_json, X_tests_json, y_train, y_tests

def train_model(model, X_train, X_tests, y_train, y_tests, epochs=10):
    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_tests, y_tests))
    return history


def get_evaluation_model(model, X_tests, y_tests):
    evaulation_dict = model.evaluate(X_tests, y_tests, verbose=2, return_dict=True)
    return evaulation_dict

def true_positives(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.int32)  # se binariza la predicción
    y_true_binary = tf.cast(y_true, dtype=tf.int32)  # se binariza la etiqueta verdadera
    true_positives = tf.reduce_sum(y_true_binary * y_pred_binary)  # se calculan los verdaderos positivos
    return true_positives.numpy()

def false_positives(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.int32)  # se binariza la predicción
    y_true_binary = tf.cast(y_true, dtype=tf.int32)  # se binariza la etiqueta verdadera
    false_positives = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)  # se calculan los falsos positivos
    return false_positives.numpy()

def true_negatives(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.int32)  # se binariza la predicción
    y_true_binary = tf.cast(y_true, dtype=tf.int32)  # se binariza la etiqueta verdadera
    true_negatives = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary))  # se calculan los verdaderos negativos
    return true_negatives.numpy()

def false_negatives(y_true, y_pred, threshold=0.5):
    y_pred_binary = tf.cast(y_pred >= threshold, dtype=tf.int32)  # se binariza la predicción
    y_true_binary = tf.cast(y_true, dtype=tf.int32)  # se binariza la etiqueta verdadera
    false_negatives = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))  # se calculan los falsos negativos
    return false_negatives.numpy()

def tp_threshold(threshold=0.5):
    def threshold_fn(y_true, y_pred):
        y_pred = tf.where(y_pred >= threshold, 1, 0)
        return y_true, y_pred

    tp = tf.keras.metrics.TruePositives(name="tp", thresholds=threshold_fn)
    return tp

def tn_threshold(threshold=0.5):
    def threshold_fn(y_true, y_pred):
        y_pred = tf.where(y_pred >= threshold, 1, 0)
        return y_true, y_pred

    tn = tf.keras.metrics.TrueNegatives(name="tn", thresholds=threshold_fn)
    return tn

def fp_threshold(threshold=0.5):
    def threshold_fn(y_true, y_pred):
        y_pred = tf.where(y_pred >= threshold, 1, 0)
        return y_true, y_pred

    fp = tf.keras.metrics.FalsePositives(name="fp", thresholds=threshold_fn)
    return fp

def fn_threshold(threshold=0.5):
    def threshold_fn(y_true, y_pred):
        y_pred = tf.where(y_pred >= threshold, 1, 0)
        return y_true, y_pred

    fn = tf.keras.metrics.FalseNegatives(name="fn", thresholds=threshold_fn)
    return fn

def recall_threshold(threshold=0.5):

    def threshold_fn(y_true, y_pred):
        y_pred = tf.where(y_pred >= threshold, 1, 0)
        return y_true, y_pred

    def recall(y_true, y_pred):
        y_true, y_pred = threshold_fn(y_true, y_pred)
        return recall_score(y_true.numpy(), y_pred.numpy())

    return recall

def precision_threshold(threshold=0.5):
    def threshold_fn(y_true, y_pred):
        y_pred = tf.where(y_pred >= threshold, 1, 0)
        return y_true, y_pred

    def precision(y_true, y_pred):
        y_true, y_pred = threshold_fn(y_true, y_pred)
        return precision_score(y_true.numpy(), y_pred.numpy())
    return precision


def f1_score_threshold(threshold=0.5):
    def threshold_fn(y_true, y_pred):
        y_pred = tf.where(y_pred >= threshold, 1, 0)
        return y_true, y_pred

    def f1_score(y_true, y_pred):
        y_true, y_pred = threshold_fn(y_true, y_pred)
        return f1_score(y_true.numpy(), y_pred.numpy())
    return f1_score

def accuracy_threshold(threshold=0.5):
    def accuracy(y_true, y_pred):
        tp = tf.cast(true_positives(y_true, y_pred, threshold), dtype=tf.float32)
        tn = tf.cast(true_negatives(y_true, y_pred, threshold), dtype=tf.float32)
        fp = tf.cast(false_positives(y_true, y_pred, threshold), dtype=tf.float32)
        fn = tf.cast(false_negatives(y_true, y_pred, threshold), dtype=tf.float32)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy
    return accuracy

def save_model(model, fullpath):
    model.save(fullpath)


def load_model(fullpath):
    custom_objects = {"recall": recall, "precision": precision, "f1_score": f1_score}
    return models.load_model(fullpath, custom_objects=custom_objects)

