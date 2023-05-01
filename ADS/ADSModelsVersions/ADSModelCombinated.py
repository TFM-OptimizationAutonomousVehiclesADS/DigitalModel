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

class ADSModelCombinated(ADSModelAbstract):

    def __init__(self, iter_retraining=None):
        super().__init__(iter_retraining=iter_retraining)
        self.modelPath = os.path.join(getModelPath(), "actual_model.h5")
        self.models_configs = json.loads(os.environ.get('DIGITAL_MODEL_COMBINE_MODEL_CONFIGS'))
        self.model = self.__load_model__()
        self.modelName = "Combinated_" + "_".join([str(x["config"]["name"]) for x in self.models_configs])
        if iter_retraining:
            self.modelName = self.modelName + "-Retraining" + str(iter_retraining)

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
            if retrainWeights:
                model = self.__load_model__()
            else:
                # model = keras.Model.from_config(best_hps)
                model = self.create_model_layers(self.sizeImage, 3)
                optimizer = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_OPTIMIZER", "adam")
                threshold = self.threshold
                metrics = self.metrics
                self.compile_model(model, optimizer, metrics)

        return model, tuner

    def compile_model(self, model, optimizer="adam", metrics=["accuracy"], tunning=False, hp=None):
        model.compile(optimizer=getModelCompilerOptimizer(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=metrics)

    def create_model_layers(self, size_image, number_features):
        models = []
        index_model = 0

        for model_config in self.models_configs:
            model = tf.keras.models.model_from_json(json.dumps(model_config))
            # Cambiar los nombres de las capas del modelo para que sean unicos
            for layer in model.layers:
                layer._name = layer.name + '_' + str(index_model)
            models.append(model)
            index_model += 1

        # OUTPUT LAYER
        combined_outputs = tf.keras.layers.concatenate([model.output for model in models])
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(combined_outputs)

        model = tf.keras.Model(inputs=[model.input for model in models], outputs=output_layer, name=self.modelName)
        model.summary()
        return model

    def retrain_model(self, retraining=True, retrainWeights=False, tunning=False, model_by_best_epoch=False, random=None,
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

    def __preprocessing_X__(self, X):
        X_full_images = np.array(list(list(zip(*X))[0]))
        X_objects_images = np.array(list(list(zip(*X))[1]))
        X_surfaces_images = np.array(list(list(zip(*X))[2]))
        X_features = np.array(list(list(zip(*X))[3]))
        index_model = 0
        X_json = {}
        for model_config in self.models_configs:
            X_json["full_images_" + str(index_model)] = X_full_images
            X_json["objects_images" + str(index_model)] = X_objects_images
            X_json["surfaces_images" + str(index_model)] = X_surfaces_images
            X_json["features" + str(index_model)] = X_features
        return X_json
