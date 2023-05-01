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
import string
from ADS.ADSModelAbstract import ADSModelAbstract

class ADSModelMultiple(ADSModelAbstract):

    def __init__(self, iter_retraining=None):
        super().__init__(iter_retraining=iter_retraining)
        self.models_configs = json.loads(os.environ.get('DIGITAL_MODEL_COMBINE_MODEL_CONFIGS'))
        self.models = self.__load_model__()

    def __load_model__(self):
        models = []
        custom_objects = {"recall": recall_threshold(self.threshold), "precision": precision_threshold(self.threshold),
                          "f1_score": f1_score_threshold(self.threshold), "tp": tp_threshold(self.threshold),
                          "tn": tn_threshold(self.threshold),
                          "fp": fp_threshold(self.threshold), "fn": fn_threshold(self.threshold)}
        index_model = 0
        for model_config in self.models_configs:
            model_path = os.path.join(getModelPath(), "actual_model_" + str(index_model) + ".h5")
            if os.path.exists(model_path):
                custom_objects = {"recall": recall_threshold(self.threshold),
                                  "precision": precision_threshold(self.threshold),
                                  "f1_score": f1_score_threshold(self.threshold), "tp": tp_threshold(self.threshold),
                                  "tn": tn_threshold(self.threshold),
                                  "fp": fp_threshold(self.threshold), "fn": fn_threshold(self.threshold)}
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                models.append(model)
            index_model += 1

        if len(models) == 0:
            models = None

        return models

    def get_metric_result_combination(self, values, result_by="mean"):
        result = 0
        if result_by == "mean":
            result = sum(values) / len(values)
        elif result_by == "max":
            result = max(values)
        elif result_by == "min":
            result = min(values)
        return result

    def get_evaluation_model(self, X, y):
        evaluations_dict = []
        evaluation_dict_values = {}
        for model in self.models:
            evaluation_dict = self.model.evaluate(X, y, verbose=2, return_dict=True)
            evaluations_dict.append(evaluations_dict)
        evaluation_dict_final = self.__get_evaluation_dict_by_evaluations__(evaluations_dict)
        return evaluation_dict_final

    def __get_evaluation_dict_by_evaluations__(self, evaluations_dict):
        evaluation_dict_final = {}
        evaluation_dict_values = {}

        for evaluation_dict in evaluations_dict:
            for key, value in evaluation_dict.items():
                if key not in evaluation_dict_values:
                    evaluation_dict_values[key] = []
                evaluation_dict_values[key].append(value)

        for key, values in evaluation_dict_values.items():
            if len(values) > 0 and all(isinstance(item, (int, float)) for item in values):
                evaluation_dict_final[key] = self.get_metric_result_combination(values)
            elif len(values) > 0:
                evaluation_dict_final[key] = values[0]

        return evaluation_dict_final

    def get_predict_result_combination(self, values, result_by="mean"):
        result = 0
        if result_by == "mean":
            result = sum(values) / len(values)
        elif result_by == "max":
            result = max(values)
        elif result_by == "min":
            result = min(values)
        return result

    def predict_sample(self, sample):
        X, y = self.__preprocessing_sample__(sample)
        X = self.__preprocessing_X__(X)
        y = self.__preprocessing_y__(y)

        yhats = []
        for model in self.models:
            yhats.append(float(model.predict(X)[0][0]))

        result_by = os.environ.get('DIGITAL_MODEL_COMBINE_MODEL_RESULT_BY', "mean")
        yhat = self.get_predict_result_combination(yhats, result_by=result_by)

        if float(yhat) > self.thresholdHigh:
            sample["prediction"] = float(yhat)
            self.add_sample_to_high_anomalies_dataset(sample)

        if "anomaly" in sample and int(sample["anomaly"] == 1):
            self.add_sample_to_dataset(sample, reviewed=True)

        return yhat

    def save_model(self, model, modelPath=None):
        model.save(modelPath)

    def get_actual_model_json(self):
        result_json = []
        for model in self.models:
            result_json.append(model.to_json())
        return result_json

    def random_string(self, length=10):
        chars = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(chars) for _ in range(length))
        return random_string

    def get_model_image_base64(self, model):
        model_image_path = getImageModelPath()
        model_image = keras.utils.plot_model(model, to_file=model_image_path, show_shapes=True)
        with open(model_image_path, 'rb') as f:
            img_bytes = io.BytesIO(f.read())
        # Convertir la imagen a base64
        base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return base64_image

    def get_model_image_base64_multiple_models(self, models):
        model_image_paths = []
        for model in models:
            model_image_path = getModelPath() + self.random_string() + ".png"
            model_image = keras.utils.plot_model(model, to_file=model_image_path, show_shapes=True)
            model_image_paths.append(model_image_path)

        model_images = []
        model_images_height = []
        model_images_width = []
        for model_image_path in model_image_paths:
            model_image = Image.open(model_image_path)
            model_images.append(model_image)
            model_images_height.append(model_image.height)
            model_images_width.append(model_image.width)

        max_height = max(model_images_height)
        sum_width = sum(model_images_width)
        for model_image in model_images:
            model_image = model_image.resize(
                (int(model_image.width * (max_height / model_image.height)), max_height))

        # Crea la imagen de salida
        output_image = Image.new('RGB', (sum_width, max_height))

        start_width = 0
        for model_image in model_images:
            output_image.paste(model_image, (start_width, 0))
            start_width += model_image.width

        output_image_path = getModelPath() + self.random_string() + ".png"
        output_image.save(output_image_path)

        with open(output_image_path, 'rb') as f:
            img_bytes = io.BytesIO(f.read())

        # Convertir la imagen a base64
        base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        return base64_image

    def __get_model_by_config__(self, retraining, retrainWeights, tunning, X_train, y_train, epochs, validation_split,
                                tuner=None, index_model=0):
        metric_objective = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_METRIC_OBJECTIVE", "f1_score")
        model = None

        if not retraining:
            model = self.create_model_layers(self.sizeImage, 3, index_model=index_model)
            optimizer = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_OPTIMIZER", "adam")
            threshold = self.threshold
            metrics = self.metrics
            self.compile_model(model, optimizer, metrics)

        else:
            if retrainWeights:
                model = self.__load_model__()
            else:
                model = self.create_model_layers(self.sizeImage, 3, index_model=index_model)
                optimizer = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_OPTIMIZER", "adam")
                threshold = self.threshold
                metrics = self.metrics
                self.compile_model(model, optimizer, metrics)
        print(model)
        return model, tuner

    def compile_model(self, model, optimizer="adam", metrics=["accuracy"], tunning=False, hp=None):
        model.compile(optimizer=getModelCompilerOptimizer(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=metrics)

    def create_model_layers(self, size_image, number_features, index_model=0):
        custom_objects = {"recall": recall_threshold(self.threshold), "precision": precision_threshold(self.threshold),
                          "f1_score": f1_score_threshold(self.threshold), "tp": tp_threshold(self.threshold),
                          "tn": tn_threshold(self.threshold),
                          "fp": fp_threshold(self.threshold), "fn": fn_threshold(self.threshold)}
        model_config = self.models_configs[index_model]
        model = tf.keras.models.model_from_config(model_config, custom_objects=custom_objects)
        model.summary()
        return model

    def retrain_model(self, retraining=True, retrainWeights=False, tunning=False, model_by_best_epoch=False, random=None,
                      size_split=None, test_size=0.25, epochs=10, validation_split=0.2):
        X_train, X_tests, y_train, y_tests = self.__get_train_test_split__(self.dataset, random, size_split, test_size)

        metric_objective = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_METRIC_OBJECTIVE", "f1_score")

        evaluations_dict = []
        models_retrained = []

        index_model = 0
        for model_config in self.models_configs:
            model, tuner = self.__get_model_by_config__(retraining, retrainWeights, tunning, X_train, y_train, epochs,
                                                        validation_split, index_model=index_model)

            # RETRAIN
            history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split)

            if model_by_best_epoch:
                val_metric_per_epoch = history.history['val_' + metric_objective]
                best_epoch = val_metric_per_epoch.index(max(val_metric_per_epoch)) + 1
                logging.info('Best epoch: %d' % (best_epoch,))
                model, tuner = self.__get_model_by_config__(retraining, retrainWeights, tunning, X_train, y_train, epochs,
                                                            validation_split, tuner=tuner, index_model=index_model)
                # RETRAIN WITH BEST EPOCH
                history = model.fit(X_train, y_train, epochs=best_epoch, validation_split=validation_split)

            evaluation_dict = self.get_evaluation_dict(model, X_tests, y_tests)
            evaluation_dict["best_epoch"] = model_by_best_epoch
            evaluation_dict["epochs"] = epochs
            evaluation_dict["model_config"] = model.get_config()
            evaluations_dict.append(evaluation_dict)
            models_retrained.append(model)
            addNewRetrainingEvaluation(evaluation_dict)

            index_model += 1

        evaluation_dict_final = self.__get_evaluation_dict_by_evaluations__(evaluations_dict)
        evaluation_dict_final["model_image_base64"] = self.get_model_image_base64_multiple_models(models_retrained)
        evaluation_dict_final["nameRetrainingModel"] = self.modelName
        evaluation_dict_final["retraining"] = retraining
        evaluation_dict_final["test_size"] = test_size
        evaluation_dict_final["size_split"] = size_split
        evaluation_dict_final["tunning"] = tunning
        evaluation_dict_final["random"] = random
        evaluation_dict_final["retrain_weights"] = retrainWeights
        if "_id" in evaluation_dict_final:
            del evaluation_dict_final["_id"]

        print(evaluation_dict_final)
        self.save_evaluation_model(evaluation_dict_final)

        best_model_found = self.is_better_model(evaluation_dict_final, metric=metric_objective)
        if not retraining or best_model_found:
            logging.info("RETRAINED AND SAVING - NEW BEST MODEL")
            index_model = 0
            for model in models_retrained:
                model_path = os.path.join(getModelPath(), "actual_model_" + str(index_model) + ".h5")
                self.save_model(model, modelPath=model_path)
                index_model += 1
        return best_model_found