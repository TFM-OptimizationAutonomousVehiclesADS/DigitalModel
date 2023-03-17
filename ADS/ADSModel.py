import os.path

from localDatabase.collections.MLModelConfiguration.queries import *
from localDatabase.collections.SamplesDataset.queries import *
from localDatabase.collections.TrainingEvaluationLogs.queries import addNewRetrainingEvaluation
import pandas as pd
from ADS.auxiliar_functions import *
import tensorflow as tf
from tensorflow import keras
import json
import logging
import io
import base64
import os
import keras_tuner as kt


class ADSModel:

    def __init__(self):
        self.modelPath = getModelPath()
        self.resizedImagesPath = getPathResizedImage()
        self.objectsImagesPath = getPathObjectsImage()
        self.surfacesImagesPath = getPathSurfacesImage()
        self.model = self.__load_model__()
        self.sizeImage = getSizeImage()
        self.threshold = getThreshold()
        self.pathDatasetCsv = getPathDatasetCsv()
        self.dataset = pd.read_csv(self.pathDatasetCsv)

    def __load_model__(self):
        if os.path.exists(self.modelPath):
            custom_objects = {"recall": recall, "precision": precision, "f1_score": f1_score}
            return models.load_model(self.modelPath, custom_objects=custom_objects)
        return None

    def __preprocessing_X__(self, X):
        X_full_images = np.array(list(list(zip(*X))[0]))
        X_objects_images = np.array(list(list(zip(*X))[1]))
        X_surfaces_images = np.array(list(list(zip(*X))[2]))
        X_features = np.array(list(list(zip(*X))[3]))
        X_json = {"full_images": X_full_images,
                  "objects_images": X_objects_images,
                  "surfaces_images": X_surfaces_images,
                  "features": X_features
                  }
        return X_json

    def __preprocessing_y__(self, y):
        y = np.array(y)
        return y

    def __preprocessing_sample__(self, sample):
        X = []
        y = []

        key_camera_token = sample["key_camera_token"]
        filename = key_camera_token + ".jpg"

        im1 = np.array(
            resize_image(Image.open(self.resizedImagesPath + "/" + filename), size_image=self.sizeImage)) / 255.0
        im2 = np.array(
            resize_image(Image.open(self.objectsImagesPath + "/" + filename), size_image=self.sizeImage)) / 255.0
        im3 = np.array(
            resize_image(Image.open(self.surfacesImagesPath + "/" + filename), size_image=self.sizeImage)) / 255.0
        ims_array = [im1, im2, im3]

        camera = get_float_channel_camera(sample["channel_camera"])
        speed = sample["speed"]
        rotation = sample["rotation_rate_z"]
        features_array = [camera, speed, rotation]

        X.append([im1, im2, im3, features_array])
        y.append([sample["anomaly"]])

        return X, y

    def __preprocessing_dataframe__(self, df):
        X = []
        y = []
        for index, row in df.iterrows():
            X_sample, y_sample = self.__preprocessing_sample__(row.to_dict())
            X.extend(X_sample)
            y.extend(y_sample)
        return X, y

    def get_evaluation_model(self, X, y):
        evaulation_dict = self.model.evaluate(X, y, verbose=2, return_dict=True)
        return evaulation_dict

    def predict_sample(self, sample):
        X, y = self.__preprocessing_sample__(sample)
        X = self.__preprocessing_X__(X)
        y = self.__preprocessing_y__(y)
        yhat = self.model.predict(X)
        yhat = float(yhat[0][0])
        return yhat

    def is_anomaly(self, y_pred):
        return y_pred >= self.threshold

    def is_sure_normal_sample(self, y_pred):
        return y_pred <= 0.1

    def __get_train_test_split__(self, random=None, size_split=None, test_size=0.25):
        dataset = self.dataset
        if random:
            dataset = dataset.sample(frac=1)
        if size_split:
            dataset = dataset.head(size_split)
        X, y = self.__preprocessing_dataframe__(dataset)
        X_train, X_tests, y_train, y_tests = train_test_split(X, y, test_size=test_size)
        X_train_json = self.__preprocessing_X__(X_train)
        X_tests_json = self.__preprocessing_X__(X_tests)
        y_train = self.__preprocessing_y__(y_train)
        y_tests = self.__preprocessing_y__(y_tests)
        return X_train_json, X_tests_json, y_train, y_tests

    def get_evaluation_dict(self, model, X_tests, y_tests):
        evaulation_dict = get_evaluation_model(model, X_tests, y_tests)
        # Obtener las predicciones del modelo en el conjunto de test
        # y_pred = model.predict(X_test)
        # y_pred = np.argmax(y_pred, axis=1)
        return evaulation_dict

    def save_model(self, model):
        fullpath = self.modelPath
        save_model(model, fullpath)

    def is_better_model(self, evaluation_dict, metric="f1_score"):
        actual_evaluation_dict = self.get_actual_evaluation_model()
        metric_obtained = int(evaluation_dict.get(metric, 0))
        metric_actual = int(actual_evaluation_dict.get(metric, 0))
        return metric_obtained >= metric_actual

    def get_actual_evaluation_model(self):
        fullpath = getEvaluationModelPath()
        with open(fullpath, 'r') as file:
            evaluation_dict = json.load(file)
        return evaluation_dict

    def save_evaluation_model(self, evaluation_dict):
        fullpath = getEvaluationModelPath()
        with open(fullpath, 'w') as file:
            file.write(json.dumps(evaluation_dict, indent=4))

    def get_model_image_base64(self, model):
        model_image_path = getImageModelPath()
        model_image = keras.utils.plot_model(model, to_file=model_image_path, show_shapes=True)
        with open(model_image_path, 'rb') as f:
            img_bytes = io.BytesIO(f.read())
        # Convertir la imagen a base64
        base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return base64_image

    def retrain_model(self, retrainWeights=True, tunning=False, model_by_best_epoch=False, random=None, size_split=None, test_size=0.25, epochs=10,
                      validation_split=0.2):
        X_train, X_tests, y_train, y_tests = self.__get_train_test_split__(random, size_split, test_size)

        model = self.model
        if tunning:
            tuner = kt.Hyperband(self.build_model_tunning,
                                 objective=kt.Objective("val_f1_score", direction="max"),
                                 max_epochs=epochs,
                                 factor=3,
                                 directory='models',
                                 project_name='tunning_model')
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            tuner.search(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])
            # Get the optimal hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)

        else:
            best_hps = self.model.get_config()
            if retrainWeights:
                model = self.__load_model__()
            else:
                model = keras.Model.from_config(best_hps)

        history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
        if model_by_best_epoch:
            # val_acc_per_epoch = history.history['val_accuracy']
            val_f1_score_per_epoch = history.history['val_f1_score']
            best_epoch = val_f1_score_per_epoch.index(max(val_f1_score_per_epoch)) + 1
            logging.info('Best epoch: %d' % (best_epoch,))
            if tunning:
                model = tuner.hypermodel.build(best_hps)
            else:
                if retrainWeights:
                    model = self.__load_model__()
                else:
                    model = keras.Model.from_config(best_hps)
            history = model.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)

        evaluation_dict = self.get_evaluation_dict(model, X_tests, y_tests)
        evaluation_dict["random"] = random
        evaluation_dict["size_split"] = size_split
        evaluation_dict["test_size"] = test_size
        evaluation_dict["epochs"] = epochs
        evaluation_dict["retraining"] = True
        evaluation_dict["model_config"] = self.model.get_config()
        evaluation_dict["model_image_base64"] = self.get_model_image_base64(model)
        if self.is_better_model(evaluation_dict):
            logging.info("RETRAINED AND SAVING - NEW BEST MODEL")
            self.save_model(model)
            self.save_evaluation_model(evaluation_dict)
            self.model = model
        addNewRetrainingEvaluation(evaluation_dict)

    def train_new_model(self, random=None, size_split=None, test_size=0.25, epochs=10):
        model = self.create_model_layers(self.sizeImage, 3)
        optimizer = getModelCompilerOptimizer()
        metrics = ["accuracy", f1_score, recall, precision]
        self.compile_model(model, optimizer, metrics)
        X_train, X_tests, y_train, y_tests = self.__get_train_test_split__(random, size_split, test_size)
        history = model.fit(X_train, y_train, epochs=epochs)
        evaluation_dict = self.get_evaluation_dict(model, X_tests, y_tests)
        evaluation_dict["random"] = random
        evaluation_dict["size_spit"] = size_split
        evaluation_dict["test_size"] = test_size
        evaluation_dict["epochs"] = epochs
        evaluation_dict["optimizer"] = optimizer
        evaluation_dict["retraining"] = False
        evaluation_dict["model_config"] = model.get_config()
        evaluation_dict["model_image_base64"] = self.get_model_image_base64(model)
        self.save_model(model)
        self.save_evaluation_model(evaluation_dict)
        self.model = model
        addNewRetrainingEvaluation(evaluation_dict)

    def compile_model(self, model, optimizer="adam", metrics=["accuracy"], tunning=False, hp=None):
        if tunning:
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                          loss=keras.losses.BinaryCrossentropy(),
                          metrics=metrics)
        else:
            model.compile(optimizer=getModelCompilerOptimizer(),
                          loss=keras.losses.BinaryCrossentropy(),
                          metrics=metrics)

    def create_model_layers(self, size_image, number_features):
        # Process features input layer
        input_features = layers.Input(shape=(number_features,), name="features")
        dense_features = layers.Dense(64)(input_features)

        # Process image input layer
        input_full_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="full_images")
        input_full_conv2d1 = layers.Conv2D(16, (3, 3), activation='relu')(input_full_images)
        input_full_pooling2d1 = layers.MaxPooling2D(2, 2)(input_full_conv2d1)
        # input_full_conv2d2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_full_pooling2d1)
        # input_full_pooling2d2 = layers.MaxPooling2D(2, 2)(input_full_conv2d2)
        # input_full_conv2d3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_full_pooling2d2)
        # input_full_pooling2d3 = layers.MaxPooling2D(2, 2)(input_full_conv2d3)
        input_full_flatten_images = layers.Flatten()(input_full_pooling2d1)
        input_full_dense_images = layers.Dense(64)(input_full_flatten_images)

        # Process objects_image input layer
        input_objects_image = layers.Input(shape=(size_image[0], size_image[1], 3), name="objects_images")
        input_objects_conv2d1 = layers.Conv2D(16, (3, 3), activation='relu')(input_objects_image)
        input_objects_pooling2d1 = layers.MaxPooling2D(2, 2)(input_objects_conv2d1)
        # input_objects_conv2d2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_objects_pooling2d1)
        # input_objects_pooling2d2 = layers.MaxPooling2D(2, 2)(input_objects_conv2d2)
        # input_objects_conv2d3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_objects_pooling2d2)
        # input_objects_pooling2d3 = layers.MaxPooling2D(2, 2)(input_objects_conv2d3)
        input_objects_flatten_images = layers.Flatten()(input_objects_pooling2d1)
        input_objects_dense_images = layers.Dense(64)(input_objects_flatten_images)

        # Process surfaces input layer
        input_surfaces_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="surfaces_images")
        input_surfaces_conv2d1 = layers.Conv2D(16, (3, 3), activation='relu')(input_surfaces_images)
        input_surfaces_pooling2d1 = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d1)
        # input_surfaces_conv2d2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_surfaces_pooling2d1)
        # input_surfaces_pooling2d2 = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d2)
        # input_surfaces_conv2d3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_surfaces_pooling2d2)
        # input_surfaces_pooling2d3 = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d3)
        input_surfaces_flatten_images = layers.Flatten()(input_surfaces_pooling2d1)
        input_surfaces_dense_images = layers.Dense(64)(input_surfaces_flatten_images)

        # embedding_images = layers.Embedding(32, 64)(dense_images)

        # Concatenate both inputs layers
        x = layers.concatenate(
            [input_features, input_full_flatten_images, input_objects_flatten_images, input_surfaces_flatten_images])

        # Output layer
        output_layer = layers.Dense(1, activation="sigmoid", name="output")(x)

        # Model
        model = keras.Model(inputs=[input_features, input_full_images, input_objects_image, input_surfaces_images],
                            outputs=[output_layer])
        model.summary()
        return model

    def create_model_layers_tunning(self, size_image, number_features, hp):
        # Process features input layer
        input_features = layers.Input(shape=(number_features,), name="features")
        dense_features = layers.Dense(units=hp.Int(name='units', min_value=16, max_value=256, step=32))(input_features)

        # Process image input layer
        input_full_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="full_images")
        # input_full_conv2d1 = layers.Conv2D(16, (3, 3), activation='relu')(input_full_images)
        # input_full_pooling2d1 = layers.MaxPooling2D(2, 2)(input_full_conv2d1)
        # input_full_conv2d2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_full_pooling2d1)
        # input_full_pooling2d2 = layers.MaxPooling2D(2, 2)(input_full_conv2d2)
        # input_full_conv2d3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_full_pooling2d2)
        # input_full_pooling2d3 = layers.MaxPooling2D(2, 2)(input_full_conv2d3)
        input_full_flatten_images = layers.Flatten()(input_full_images)
        input_full_dense_images = layers.Dense(hp.Int(name='units', min_value=16, max_value=256, step=32))(
            input_full_flatten_images)

        # Process objects_image input layer
        input_objects_image = layers.Input(shape=(size_image[0], size_image[1], 3), name="objects_images")
        # input_objects_conv2d1 = layers.Conv2D(16, (3, 3), activation='relu')(input_objects_image)
        # input_objects_pooling2d1 = layers.MaxPooling2D(2, 2)(input_objects_conv2d1)
        # input_objects_conv2d2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_objects_pooling2d1)
        # input_objects_pooling2d2 = layers.MaxPooling2D(2, 2)(input_objects_conv2d2)
        # input_objects_conv2d3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_objects_pooling2d2)
        # input_objects_pooling2d3 = layers.MaxPooling2D(2, 2)(input_objects_conv2d3)
        input_objects_flatten_images = layers.Flatten()(input_objects_image)
        input_objects_dense_images = layers.Dense(hp.Int(name='units', min_value=16, max_value=256, step=32))(
            input_objects_flatten_images)

        # Process surfaces input layer
        input_surfaces_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="surfaces_images")
        # input_surfaces_conv2d1 = layers.Conv2D(16, (3, 3), activation='relu')(input_surfaces_images)
        # input_surfaces_pooling2d1 = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d1)
        # input_surfaces_conv2d2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_surfaces_pooling2d1)
        # input_surfaces_pooling2d2 = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d2)
        # input_surfaces_conv2d3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_surfaces_pooling2d2)
        # input_surfaces_pooling2d3 = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d3)
        input_surfaces_flatten_images = layers.Flatten()(input_surfaces_images)
        input_surfaces_dense_images = layers.Dense(hp.Int(name='units', min_value=16, max_value=256, step=32))(
            input_surfaces_flatten_images)

        # embedding_images = layers.Embedding(32, 64)(dense_images)

        # Concatenate both inputs layers
        x = layers.concatenate(
            [dense_features, input_full_dense_images, input_objects_dense_images, input_surfaces_dense_images])

        # Output layer
        output_layer = layers.Dense(1, activation="sigmoid", name="output")(x)

        # Model
        model = keras.Model(inputs=[input_features, input_full_images, input_objects_image, input_surfaces_images],
                            outputs=[output_layer])
        model.summary()
        return model

    def build_model_tunning(self, hp):
        model = self.create_model_layers_tunning(self.sizeImage, 3, hp)
        optimizer = getModelCompilerOptimizer()
        metrics = ["accuracy", f1_score, recall, precision]
        self.compile_model(model, optimizer, metrics, tunning=True, hp=hp)
        return model