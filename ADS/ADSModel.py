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


class ADSModel:

    def __init__(self):
        self.modelPath = getModelPath()
        self.resizedImagesPath = getPathResizedImage()
        self.objectsImagesPath = getPathObjectsImage()
        self.surfacesImagesPath = getPathSurfacesImage()
        self.model = self.__load_model__()
        widthImage = int(os.environ.get('DIGITAL_MODEL_SIZE_IMAGES_WIDTH', 80))
        heightImage = int(os.environ.get('DIGITAL_MODEL_SIZE_IMAGES_HEIGHT', 45))
        self.sizeImage = [heightImage, widthImage]
        self.threshold = float(os.environ.get('DIGITAL_MODEL_THRESHOLD_ANOMALY', 0.5))
        self.thresholdHigh = float(os.environ.get('DIGITAL_MODEL_HIGH_THRESHOLD_ANOMALY', 0.85))
        self.pathDatasetCsv = getPathDatasetCsv()
        self.pathDatasetReviewedCsv = getPathDatasetReviewedCsv()
        self.pathDatasetHighAnomaliesCsv = getPathDatasetHighAnomaliesCsv()
        self.dataset = pd.read_csv(self.pathDatasetCsv)
        self.__load_dataset_reviewed__()
        self.__load_dataset_high_anomalies__()
        self.metrics = [accuracy_threshold(self.threshold), f1_score_threshold(self.threshold),
                        recall_threshold(self.threshold), precision_threshold(self.threshold),
                        tp_threshold(self.threshold), tn_threshold(self.threshold),
                        fp_threshold(self.threshold), fn_threshold(self.threshold)]


    def __load_dataset_reviewed__(self):
        if os.path.exists(self.pathDatasetReviewedCsv):
            self.datasetReviewed = pd.read_csv(self.pathDatasetReviewedCsv)
        else:
            self.datasetReviewed = pd.DataFrame(columns=self.dataset.columns)
            self.datasetReviewed.to_csv(self.pathDatasetReviewedCsv, index=False)

    def __load_dataset_high_anomalies__(self):
        if os.path.exists(self.pathDatasetHighAnomaliesCsv):
            self.datasetHighAnomalies = pd.read_csv(self.pathDatasetHighAnomaliesCsv)
        else:
            self.datasetHighAnomalies = pd.DataFrame(columns=self.dataset.columns)
            self.datasetHighAnomalies.to_csv(self.pathDatasetHighAnomaliesCsv, index=False)

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

    def get_resized_images_base64(self, sample):
        key_camera_token = sample["key_camera_token"]
        filename = key_camera_token + ".jpg"

        im1 = resize_image(Image.open(self.resizedImagesPath + "/" + filename), size_image=self.sizeImage)
        buffer = io.BytesIO()
        im1.save(buffer, format='PNG')
        im1 = buffer.getvalue()
        im1 = base64.b64encode(im1).decode('utf-8')
        im2 = resize_image(Image.open(self.objectsImagesPath + "/" + filename), size_image=self.sizeImage)
        buffer = io.BytesIO()
        im2.save(buffer, format='PNG')
        im2 = buffer.getvalue()
        im2 = base64.b64encode(im2).decode('utf-8')
        im3 = resize_image(Image.open(self.surfacesImagesPath + "/" + filename), size_image=self.sizeImage)
        buffer = io.BytesIO()
        im3.save(buffer, format='PNG')
        im3 = buffer.getvalue()
        im3 = base64.b64encode(im3).decode('utf-8')
        ims_array = [im1, im2, im3]

        return ims_array

    def __preprocessing_sample__(self, sample):
        X = []
        y = []

        if "key_camera_token" in sample:
            key_camera_token = sample["key_camera_token"]
            filename = key_camera_token + ".jpg"
            resized_image = Image.open(self.resizedImagesPath + "/" + filename)
            object_image = Image.open(self.objectsImagesPath + "/" + filename)
            surface_image = Image.open(self.surfacesImagesPath + "/" + filename)
        else:
            resized_image = Image.open(io.BytesIO(base64.b64decode(sample["resizedImage"].encode("utf-8"))))
            object_image = Image.open(io.BytesIO(base64.b64decode(sample["objectImage"].encode("utf-8"))))
            surface_image = Image.open(io.BytesIO(base64.b64decode(sample["surfaceImage"].encode("utf-8"))))

        # Convertir la imagen a RGB
        resized_image = resized_image.convert('RGB')
        object_image = object_image.convert('RGB')
        surface_image = surface_image.convert('RGB')

        im1 = np.array(
            resize_image(resized_image, size_image=self.sizeImage)) / 255.0
        im2 = np.array(
            resize_image(object_image, size_image=self.sizeImage)) / 255.0
        im3 = np.array(
            resize_image(surface_image, size_image=self.sizeImage)) / 255.0

        camera = int(get_float_channel_camera(sample["channel_camera"]))
        speed = float(sample["speed"])
        rotation = float(sample["rotation_rate_z"])
        features_array = [camera, speed, rotation]

        X.append([im1, im2, im3, features_array])

        if "anomaly" in sample:
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

        if float(yhat) > self.thresholdHigh:
            sample["prediction"] = float(yhat)
            self.add_sample_to_high_anomalies_dataset(sample)

        if "anomaly" in sample and int(sample["anomaly"] == 1):
            self.add_sample_to_dataset(sample, reviewed=True)

        return yhat

    def is_anomaly(self, y_pred):
        return float(y_pred) >= self.threshold

    def is_sure_normal_sample(self, y_pred):
        return y_pred <= 0.1

    def __get_train_test_split__(self, dataframe, random=None, size_split=None, test_size=0.25, combine_with_reviewed_dataset=True):
        dataset = dataframe
        if random:
            dataset = dataset.sample(frac=1)
        if size_split:
            dataset = dataset.head(size_split)

        # COMBINAR DATAFRAMES
        if combine_with_reviewed_dataset:
            dataset = pd.concat([dataset, self.datasetReviewed])

        X, y = self.__preprocessing_dataframe__(dataset)
        if test_size <= 0 or test_size >= 1:
            X_train, y_train = [X, y]
            X_tests, y_tests = [X, y]
        else:
            X_train, X_tests, y_train, y_tests = train_test_split(X, y, test_size=test_size)
        X_train_json = self.__preprocessing_X__(X_train)
        X_tests_json = self.__preprocessing_X__(X_tests)
        y_train = self.__preprocessing_y__(y_train)
        y_tests = self.__preprocessing_y__(y_tests)
        return X_train_json, X_tests_json, y_train, y_tests

    def evaluate_dataframe(self, dataframe):
        X_train, X_tests, y_train, y_tests = self.__get_train_test_split__(dataframe, random=None, size_split=None, test_size=0.0)
        return self.get_evaluation_dict(self.model, X_train, y_train)

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

    def __get_model_by_config__(self, retraining, retrainWeights, tunning, X_train, y_train, epochs, validation_split, tuner=None):
        metric_objective = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_METRIC_OBJECTIVE", "f1_score")
        model = None

        if not retraining:
            model = self.create_model_layers(self.sizeImage, 3)
            optimizer = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_OPTIMIZER", "adam")
            threshold = self.threshold
            metrics = self.metrics
            self.compile_model(model, optimizer, metrics)

        else:
            if tunning:
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
                    tuner.search(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])
                    # Get the optimal hyperparameters
                    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                    model = tuner.hypermodel.build(best_hps)

            else:
                best_hps = self.model.get_config()
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


    def retrain_model(self, retraining=True, retrainWeights=True, tunning=False, model_by_best_epoch=False, random=None,
                      size_split=None, test_size=0.25, epochs=10, validation_split=0.2):
        X_train, X_tests, y_train, y_tests = self.__get_train_test_split__(self.dataset, random, size_split, test_size)

        metric_objective = os.environ.get("DIGITAL_MODEL_SIZE_IMAGES_METRIC_OBJECTIVE", "f1_score")

        model, tuner = self.__get_model_by_config__(retraining, retrainWeights, tunning, X_train, y_train, epochs, validation_split)

        # RETRAIN
        history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split)

        if model_by_best_epoch:
            val_metric_per_epoch = history.history['val_' + metric_objective]
            best_epoch = val_metric_per_epoch.index(max(val_metric_per_epoch)) + 1
            logging.info('Best epoch: %d' % (best_epoch,))
            model, tuner = self.__get_model_by_config__(retraining, retrainWeights, tunning, X_train, y_train, epochs, validation_split, tuner=tuner)
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

        if not retraining or self.is_better_model(evaluation_dict, metric=metric_objective):
            logging.info("RETRAINED AND SAVING - NEW BEST MODEL")
            self.save_model(model)
            self.save_evaluation_model(evaluation_dict)
            self.model = model

        addNewRetrainingEvaluation(evaluation_dict)

    def add_sample_to_dataset(self, sampleJson, reviewed=False):
        dataset = self.dataset
        pathDataset = self.pathDatasetCsv
        if reviewed:
            dataset = self.datasetReviewed
            pathDataset = self.pathDatasetReviewedCsv

        if "anomaly" not in sampleJson and "prediction" in sampleJson:
            anomaly = self.is_anomaly(sampleJson["prediction"])
            sampleJson["anomaly"] = int(anomaly)

        # Eliminar claves no pertenecientes al dataset
        for key in list(sampleJson.keys()):
            if key not in list(dataset.columns):
                del sampleJson[key]

        # Establecemos los campos faltantes como vacio
        for col in dataset.columns:
            if col not in sampleJson:
                sampleJson[col] = ''

        # Añadimos la fila al DataFrame
        dataset = dataset.append(sampleJson, ignore_index=True)
        dataset.to_csv(pathDataset, index=False)

        return sampleJson

    def add_sample_to_high_anomalies_dataset(self, sampleJson):
        if "prediction" in sampleJson:
            anomaly = self.is_anomaly(sampleJson["prediction"])
            sampleJson["anomaly"] = int(anomaly)

        # Eliminar claves no pertenecientes al dataset
        for key in list(sampleJson.keys()):
            if key not in list(self.datasetHighAnomalies.columns):
                del sampleJson[key]

        # Establecemos los campos faltantes como vacio
        for col in self.datasetHighAnomalies.columns:
            if col not in sampleJson:
                sampleJson[col] = ''

        # Añadimos la fila al DataFrame
        dataset = self.datasetHighAnomalies.append(sampleJson, ignore_index=True)
        dataset.to_csv(self.pathDatasetHighAnomaliesCsv, index=False)

        return sampleJson

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

        # FEATURES LAYER
        final_layer_features = None
        has_dense_features = random.random() >= 0.5
        input_features = final_layer_features = layers.Input(shape=(number_features,), name="features")
        if has_dense_features:
            dense_features = final_layer_features = layers.Dense(random.choice([16, 32, 64, 128]))(input_features)

        # RESIZED IMAGES LAYER
        has_dense_resized_images = random.random() >= 0.5
        has_conv2d1_resized_images = random.random() >= 0.5
        has_conv2d2_resized_images = random.random() >= 0.5
        has_conv2d3_resized_images = random.random() >= 0.5
        final_layer_resized_images = None

        input_full_images = final_layer_resized_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="full_images")
        if has_conv2d1_resized_images:
            input_full_conv2d1 = final_layer_resized_images = layers.Conv2D(16, (3, 3), activation='relu')(input_full_images)
            input_full_pooling2d1 = final_layer_resized_images = layers.MaxPooling2D(2, 2)(input_full_conv2d1)
            if has_conv2d2_resized_images:
                input_full_conv2d2 = final_layer_resized_images = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_full_pooling2d1)
                input_full_pooling2d2 = final_layer_resized_images = layers.MaxPooling2D(2, 2)(input_full_conv2d2)
                if has_conv2d3_resized_images:
                    input_full_conv2d3 = final_layer_resized_images = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_full_pooling2d2)
                    input_full_pooling2d3 = final_layer_resized_images = layers.MaxPooling2D(2, 2)(input_full_conv2d3)
        input_full_flatten_images = final_layer_resized_images = layers.Flatten()(final_layer_resized_images)
        if has_dense_resized_images:
            input_full_dense_images = final_layer_resized_images = layers.Dense(random.choice([16, 32, 64, 128]))(input_full_flatten_images)

        # OBJECT IMAGES LAYER
        has_dense_object_images = random.random() >= 0.5
        has_conv2d1_object_images = random.random() >= 0.5
        has_conv2d2_object_images = random.random() >= 0.5
        has_conv2d3_object_images = random.random() >= 0.5
        final_layer_object_images = None

        input_objects_image = final_layer_object_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="objects_images")
        if has_conv2d1_object_images:
            input_objects_conv2d1 = final_layer_object_images = layers.Conv2D(16, (3, 3), activation='relu')(input_objects_image)
            input_objects_pooling2d1 = final_layer_object_images = layers.MaxPooling2D(2, 2)(input_objects_conv2d1)
            if has_conv2d2_object_images:
                input_objects_conv2d2 = final_layer_object_images = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_objects_pooling2d1)
                input_objects_pooling2d2 = final_layer_object_images = layers.MaxPooling2D(2, 2)(input_objects_conv2d2)
                if has_conv2d3_object_images:
                    input_objects_conv2d3 = final_layer_object_images = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_objects_pooling2d2)
                    input_objects_pooling2d3 = final_layer_object_images = layers.MaxPooling2D(2, 2)(input_objects_conv2d3)
        input_objects_flatten_images = final_layer_object_images = layers.Flatten()(final_layer_object_images)
        if has_dense_object_images:
            input_objects_dense_images = final_layer_object_images = layers.Dense(random.choice([16, 32, 64, 128]))(input_objects_flatten_images)

        # OBJECT IMAGES LAYER
        has_dense_surface_images = random.random() >= 0.5
        has_conv2d1_surface_images = random.random() >= 0.5
        has_conv2d2_surface_images = random.random() >= 0.5
        has_conv2d3_surface_images = random.random() >= 0.5
        final_layer_surface_images = None

        input_surfaces_images = final_layer_surface_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="surfaces_images")
        if has_conv2d1_surface_images:
            input_surfaces_conv2d1 = final_layer_surface_images = layers.Conv2D(16, (3, 3), activation='relu')(input_surfaces_images)
            input_surfaces_pooling2d1 = final_layer_surface_images = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d1)
            if has_conv2d2_surface_images:
                input_surfaces_conv2d2 = final_layer_surface_images = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_surfaces_pooling2d1)
                input_surfaces_pooling2d2 = final_layer_surface_images = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d2)
                if has_conv2d3_surface_images:
                    input_surfaces_conv2d3 = final_layer_surface_images = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_surfaces_pooling2d2)
                    input_surfaces_pooling2d3 = final_layer_surface_images = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d3)
        input_surfaces_flatten_images = final_layer_surface_images = layers.Flatten()(final_layer_surface_images)
        if has_dense_surface_images:
            input_surfaces_dense_images = final_layer_surface_images = layers.Dense(random.choice([16, 32, 64, 128]))(input_surfaces_flatten_images)

        # CONCATENATE LAYER
        concatenate_layer = layers.concatenate([final_layer_features, final_layer_resized_images, final_layer_object_images, final_layer_surface_images])

        # OUTPUT LAYER
        output_layer = layers.Dense(1, activation="sigmoid", name="output")(concatenate_layer)

        # Model
        model = keras.Model(inputs=[input_features, input_full_images, input_objects_image, input_surfaces_images],
                            outputs=[output_layer])
        model.summary()
        return model

    def create_model_layers_tunning(self, size_image, number_features, hp):

        # FEATURES LAYER
        final_layer_features = None
        has_dense_features = random.random() >= 0.5
        input_features = final_layer_features = layers.Input(shape=(number_features,), name="features")
        if has_dense_features:
            dense_features = final_layer_features = layers.Dense(units=hp.Int(name='units', min_value=16, max_value=256, step=32))(input_features)

        # RESIZED IMAGES LAYER
        has_dense_resized_images = random.random() >= 0.5
        has_conv2d1_resized_images = random.random() >= 0.5
        has_conv2d2_resized_images = random.random() >= 0.5
        has_conv2d3_resized_images = random.random() >= 0.5
        final_layer_resized_images = None

        input_full_images = final_layer_resized_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="full_images")
        if has_conv2d1_resized_images:
            input_full_conv2d1 = final_layer_resized_images = layers.Conv2D(16, (3, 3), activation='relu')(input_full_images)
            input_full_pooling2d1 = final_layer_resized_images = layers.MaxPooling2D(2, 2)(input_full_conv2d1)
            if has_conv2d2_resized_images:
                input_full_conv2d2 = final_layer_resized_images = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_full_pooling2d1)
                input_full_pooling2d2 = final_layer_resized_images = layers.MaxPooling2D(2, 2)(input_full_conv2d2)
                if has_conv2d3_resized_images:
                    input_full_conv2d3 = final_layer_resized_images = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_full_pooling2d2)
                    input_full_pooling2d3 = final_layer_resized_images = layers.MaxPooling2D(2, 2)(input_full_conv2d3)
        input_full_flatten_images = final_layer_resized_images = layers.Flatten()(final_layer_resized_images)
        if has_dense_resized_images:
            input_full_dense_images = final_layer_resized_images = layers.Dense(hp.Int(name='units', min_value=16, max_value=256, step=32))(input_full_flatten_images)

        # OBJECT IMAGES LAYER
        has_dense_object_images = random.random() >= 0.5
        has_conv2d1_object_images = random.random() >= 0.5
        has_conv2d2_object_images = random.random() >= 0.5
        has_conv2d3_object_images = random.random() >= 0.5
        final_layer_object_images = None

        input_objects_image = final_layer_object_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="objects_images")
        if has_conv2d1_object_images:
            input_objects_conv2d1 = final_layer_object_images = layers.Conv2D(16, (3, 3), activation='relu')(input_objects_image)
            input_objects_pooling2d1 = final_layer_object_images = layers.MaxPooling2D(2, 2)(input_objects_conv2d1)
            if has_conv2d2_object_images:
                input_objects_conv2d2 = final_layer_object_images = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_objects_pooling2d1)
                input_objects_pooling2d2 = final_layer_object_images = layers.MaxPooling2D(2, 2)(input_objects_conv2d2)
                if has_conv2d3_object_images:
                    input_objects_conv2d3 = final_layer_object_images = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_objects_pooling2d2)
                    input_objects_pooling2d3 = final_layer_object_images = layers.MaxPooling2D(2, 2)(input_objects_conv2d3)
        input_objects_flatten_images = final_layer_object_images = layers.Flatten()(final_layer_object_images)
        if has_dense_object_images:
            input_objects_dense_images = final_layer_object_images = layers.Dense(hp.Int(name='units', min_value=16, max_value=256, step=32))(input_objects_flatten_images)

        # OBJECT IMAGES LAYER
        has_dense_surface_images = random.random() >= 0.5
        has_conv2d1_surface_images = random.random() >= 0.5
        has_conv2d2_surface_images = random.random() >= 0.5
        has_conv2d3_surface_images = random.random() >= 0.5
        final_layer_surface_images = None

        input_surfaces_images = final_layer_surface_images = layers.Input(shape=(size_image[0], size_image[1], 3), name="surfaces_images")
        if has_conv2d1_surface_images:
            input_surfaces_conv2d1 = final_layer_surface_images = layers.Conv2D(16, (3, 3), activation='relu')(input_surfaces_images)
            input_surfaces_pooling2d1 = final_layer_surface_images = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d1)
            if has_conv2d2_surface_images:
                input_surfaces_conv2d2 = final_layer_surface_images = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_surfaces_pooling2d1)
                input_surfaces_pooling2d2 = final_layer_surface_images = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d2)
                if has_conv2d3_surface_images:
                    input_surfaces_conv2d3 = final_layer_surface_images = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_surfaces_pooling2d2)
                    input_surfaces_pooling2d3 = final_layer_surface_images = layers.MaxPooling2D(2, 2)(input_surfaces_conv2d3)
        input_surfaces_flatten_images = final_layer_surface_images = layers.Flatten()(input_surfaces_images)
        if has_dense_surface_images:
            input_surfaces_dense_images = final_layer_surface_images = layers.Dense(hp.Int(name='units', min_value=16, max_value=256, step=32))(input_surfaces_flatten_images)

        # CONCATENATE LAYERS
        concatenate_layer = layers.concatenate(
            [final_layer_features, final_layer_resized_images, final_layer_object_images, final_layer_surface_images])

        # Output layer
        output_layer = layers.Dense(1, activation="sigmoid", name="output")(concatenate_layer)

        # Model
        model = keras.Model(inputs=[input_features, input_full_images, input_objects_image, input_surfaces_images],
                            outputs=[output_layer])
        model.summary()
        return model

    def build_model_tunning(self, hp):
        model = self.create_model_layers_tunning(self.sizeImage, 3, hp)
        optimizer = getModelCompilerOptimizer()
        threshold = self.threshold
        metrics = self.metrics
        self.compile_model(model, optimizer, metrics, tunning=True, hp=hp)
        return model