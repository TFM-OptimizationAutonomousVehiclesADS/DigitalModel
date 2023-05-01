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
import random
from ADS.ADSModelAbstract import ADSModelAbstract

class ADSModelTunning(ADSModelAbstract):

    def __init(self, iter_retraining=None):
        super().__init__(iter_retraining=iter_retraining)
        self.modelPath = os.path.join(getModelPath(), "actual_model.h5")
        self.model = self.__load_model__()

    def __load_model__(self):
        if os.path.exists(self.modelPath):
            custom_objects = {"recall": recall_threshold(self.threshold),
                              "precision": precision_threshold(self.threshold),
                              "f1_score": f1_score_threshold(self.threshold), "tp": tp_threshold(self.threshold),
                              "tn": tn_threshold(self.threshold),
                              "fp": fp_threshold(self.threshold), "fn": fn_threshold(self.threshold)}
            return tf.keras.models.load_model(self.modelPath, custom_objects=custom_objects)
        return None

    def get_evaluation_model(self, X, y):
        evaulation_dict = self.model.evaluate(X, y, verbose=2, return_dict=True)
        return evaulation_dict

    def predict_sample(self, sample):
        X, y = self.__preprocessing_sample__(sample)
        X = self.__preprocessing_X__(X)
        y = self.__preprocessing_y__(y)
        yhat = self.model.predict(X)
        yhat = float(yhat[0][0])

        if float(yhat) > self.thresholdHigh:
            sample["prediction"] = float(yhat)
            self.add_sample_to_high_anomalies_dataset(sample)

        if "anomaly" in sample and int(sample["anomaly"] == 1):
            self.add_sample_to_dataset(sample, reviewed=True)

        return yhat

    def save_model(self, model):
        fullpath = self.modelPath
        model.save(fullpath)

    def get_actual_model_json(self):
        return self.model.to_json()

    def replace_actual_model_json(self, model_json):
        if isinstance(model_json, str):
            model_json = json.loads(model_json)
        model = tf.keras.models.model_from_json(model_json)
        self.save_model(model)

        return self.model.to_json()

    def get_model_image_base64(self, model):
        model_image_path = getImageModelPath()
        model_image = keras.utils.plot_model(model, to_file=model_image_path, show_shapes=True)
        with open(model_image_path, 'rb') as f:
            img_bytes = io.BytesIO(f.read())
        # Convertir la imagen a base64
        base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return base64_image

    def build_model_tunning(self, hp):
        model = self.create_model_layers(self.sizeImage, 3, hp)
        optimizer = getModelCompilerOptimizer()
        threshold = self.threshold
        metrics = self.metrics
        self.compile_model(model, optimizer, metrics, tunning=True, hp=hp)
        return model

    def __get_model_by_config__(self, retraining, retrainWeights, tunning, X_train, y_train, epochs, validation_split,
                                tuner=None):
        metric_objective = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_METRIC_OBJECTIVE", "f1_score")
        model = None

        if not retraining:
            model = self.create_model_layers(self.sizeImage, 3)
            optimizer = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_OPTIMIZER", "adam")
            threshold = self.threshold
            metrics = self.metrics
            self.compile_model(model, optimizer, metrics)

        else:
            if tuner:
                # Get the optimal hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model = tuner.hypermodel.build(best_hps)
            else:
                tuner = kt.Hyperband(self.build_model_tunning,
                                     objective=kt.Objective("val_" + metric_objective, direction="max"),
                                     max_epochs=epochs,
                                     factor=3,
                                     directory='models',
                                     project_name='tunning_model')
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                tuner.search(X_train, y_train, epochs=epochs, validation_split=validation_split,
                             callbacks=[stop_early])
                # Get the optimal hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model = tuner.hypermodel.build(best_hps)

        return model, tuner

    def compile_model(self, model, optimizer="adam", metrics=["accuracy"], tunning=True, hp=None):
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=metrics)

    def create_model_layers(self, size_image, number_features, hp=None):
        # FEATURES LAYER
        final_layer_features = None
        has_dense_features = random.random() >= 0.5
        input_features = final_layer_features = tf.keras.layers.Input(shape=(number_features,), name="features")
        if has_dense_features:
            dense_features = final_layer_features = tf.keras.layers.Dense(
                units=hp.Int(name='units', min_value=16, max_value=256, step=32))(input_features)

        # RESIZED IMAGES LAYER
        has_dense_resized_images = random.random() >= 0.5
        has_conv2d1_resized_images = random.random() >= 0.5
        has_conv2d2_resized_images = random.random() >= 0.5
        has_conv2d3_resized_images = random.random() >= 0.5
        final_layer_resized_images = None

        input_full_images = final_layer_resized_images = tf.keras.layers.Input(shape=(size_image[0], size_image[1], 3),
                                                                               name="full_images")
        if has_conv2d1_resized_images:
            input_full_conv2d1 = final_layer_resized_images = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(
                input_full_images)
            input_full_pooling2d1 = final_layer_resized_images = tf.keras.layers.MaxPooling2D(2, 2)(input_full_conv2d1)
            if has_conv2d2_resized_images:
                input_full_conv2d2 = final_layer_resized_images = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(
                    input_full_pooling2d1)
                input_full_pooling2d2 = final_layer_resized_images = tf.keras.layers.MaxPooling2D(2, 2)(
                    input_full_conv2d2)
                if has_conv2d3_resized_images:
                    input_full_conv2d3 = final_layer_resized_images = tf.keras.layers.Conv2D(64, (3, 3),
                                                                                             activation='relu')(
                        input_full_pooling2d2)
                    input_full_pooling2d3 = final_layer_resized_images = tf.keras.layers.MaxPooling2D(2, 2)(
                        input_full_conv2d3)
        input_full_flatten_images = final_layer_resized_images = tf.keras.layers.Flatten()(final_layer_resized_images)
        if has_dense_resized_images:
            input_full_dense_images = final_layer_resized_images = tf.keras.layers.Dense(
                hp.Int(name='units', min_value=16, max_value=256, step=32))(input_full_flatten_images)

        # OBJECT IMAGES LAYER
        has_dense_object_images = random.random() >= 0.5
        has_conv2d1_object_images = random.random() >= 0.5
        has_conv2d2_object_images = random.random() >= 0.5
        has_conv2d3_object_images = random.random() >= 0.5
        final_layer_object_images = None

        input_objects_image = final_layer_object_images = tf.keras.layers.Input(shape=(size_image[0], size_image[1], 3),
                                                                                name="objects_images")
        if has_conv2d1_object_images:
            input_objects_conv2d1 = final_layer_object_images = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(
                input_objects_image)
            input_objects_pooling2d1 = final_layer_object_images = tf.keras.layers.MaxPooling2D(2, 2)(
                input_objects_conv2d1)
            if has_conv2d2_object_images:
                input_objects_conv2d2 = final_layer_object_images = tf.keras.layers.Conv2D(32, (3, 3),
                                                                                           activation='relu')(
                    input_objects_pooling2d1)
                input_objects_pooling2d2 = final_layer_object_images = tf.keras.layers.MaxPooling2D(2, 2)(
                    input_objects_conv2d2)
                if has_conv2d3_object_images:
                    input_objects_conv2d3 = final_layer_object_images = tf.keras.layers.Conv2D(64, (3, 3),
                                                                                               activation='relu')(
                        input_objects_pooling2d2)
                    input_objects_pooling2d3 = final_layer_object_images = tf.keras.layers.MaxPooling2D(2, 2)(
                        input_objects_conv2d3)
        input_objects_flatten_images = final_layer_object_images = tf.keras.layers.Flatten()(final_layer_object_images)
        if has_dense_object_images:
            input_objects_dense_images = final_layer_object_images = tf.keras.layers.Dense(
                hp.Int(name='units', min_value=16, max_value=256, step=32))(input_objects_flatten_images)

        # OBJECT IMAGES LAYER
        has_dense_surface_images = random.random() >= 0.5
        has_conv2d1_surface_images = random.random() >= 0.5
        has_conv2d2_surface_images = random.random() >= 0.5
        has_conv2d3_surface_images = random.random() >= 0.5
        final_layer_surface_images = None

        input_surfaces_images = final_layer_surface_images = tf.keras.layers.Input(
            shape=(size_image[0], size_image[1], 3), name="surfaces_images")
        if has_conv2d1_surface_images:
            input_surfaces_conv2d1 = final_layer_surface_images = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(
                input_surfaces_images)
            input_surfaces_pooling2d1 = final_layer_surface_images = tf.keras.layers.MaxPooling2D(2, 2)(
                input_surfaces_conv2d1)
            if has_conv2d2_surface_images:
                input_surfaces_conv2d2 = final_layer_surface_images = tf.keras.layers.Conv2D(32, (3, 3),
                                                                                             activation='relu')(
                    input_surfaces_pooling2d1)
                input_surfaces_pooling2d2 = final_layer_surface_images = tf.keras.layers.MaxPooling2D(2, 2)(
                    input_surfaces_conv2d2)
                if has_conv2d3_surface_images:
                    input_surfaces_conv2d3 = final_layer_surface_images = tf.keras.layers.Conv2D(64, (3, 3),
                                                                                                 activation='relu')(
                        input_surfaces_pooling2d2)
                    input_surfaces_pooling2d3 = final_layer_surface_images = tf.keras.layers.MaxPooling2D(2, 2)(
                        input_surfaces_conv2d3)
        input_surfaces_flatten_images = final_layer_surface_images = tf.keras.layers.Flatten()(input_surfaces_images)
        if has_dense_surface_images:
            input_surfaces_dense_images = final_layer_surface_images = tf.keras.layers.Dense(
                hp.Int(name='units', min_value=16, max_value=256, step=32))(input_surfaces_flatten_images)

        # CONCATENATE LAYERS
        concatenate_layer = tf.keras.layers.concatenate(
            [final_layer_features, final_layer_resized_images, final_layer_object_images, final_layer_surface_images])

        # Output layer
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(concatenate_layer)

        # Model
        model = tf.keras.Model(inputs=[input_features, input_full_images, input_objects_image, input_surfaces_images],
                            outputs=[output_layer], name=self.modelName)
        model.summary()
        return model

    def retrain_model(self, retraining=True, retrainWeights=True, tunning=True, model_by_best_epoch=False, random=None,
                      size_split=None, test_size=0.25, epochs=10, validation_split=0.2):
        X_train, X_tests, y_train, y_tests = self.__get_train_test_split__(self.dataset, random, size_split, test_size)

        metric_objective = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_METRIC_OBJECTIVE", "f1_score")

        model, tuner = self.__get_model_by_config__(retraining, retrainWeights, tunning, X_train, y_train, epochs,
                                                    validation_split)

        # RETRAIN
        history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split)

        if model_by_best_epoch:
            val_metric_per_epoch = history.history['val_' + metric_objective]
            best_epoch = val_metric_per_epoch.index(max(val_metric_per_epoch)) + 1
            logging.info('Best epoch: %d' % (best_epoch,))
            model, tuner = self.__get_model_by_config__(retraining, retrainWeights, tunning, X_train, y_train, epochs,
                                                        validation_split, tuner=tuner)
            # RETRAIN WITH BEST EPOCH
            history = model.fit(X_train, y_train, epochs=best_epoch, validation_split=validation_split)

        evaluation_dict = self.get_evaluation_dict(model, X_tests, y_tests)
        evaluation_dict["random"] = random
        evaluation_dict["retrain_weights"] = retrainWeights
        evaluation_dict["tunning"] = tunning
        evaluation_dict["best_epoch"] = model_by_best_epoch
        evaluation_dict["history"] = history.history
        evaluation_dict["size_split"] = size_split
        evaluation_dict["test_size"] = test_size
        evaluation_dict["epochs"] = epochs
        evaluation_dict["retraining"] = retraining
        evaluation_dict["model_config"] = model.get_config()
        evaluation_dict["model_image_base64"] = self.get_model_image_base64(model)
        evaluation_dict["nameRetrainingModel"] = self.modelName

        best_model_found = self.is_better_model(evaluation_dict, metric=metric_objective)
        if not retraining or best_model_found:
            logging.info("RETRAINED AND SAVING - NEW BEST MODEL")
            self.save_model(model)
            self.save_evaluation_model(evaluation_dict)
            self.model = model

        addNewRetrainingEvaluation(evaluation_dict)
        return best_model_found