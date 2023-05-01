import logging
import os.path

from localDatabase.collections.MLModelConfiguration.queries import *
from localDatabase.collections.SamplesDataset.queries import *
import pandas as pd
from ADS.auxiliar_functions import *
import json
import io
import base64
import os
from abc import ABC, abstractmethod


class ADSModelAbstract(ABC):

    def __init__(self, iter_retraining=None):
        self.modelName = os.environ.get("DIGITAL_MODEL_NAME", "model")
        if iter_retraining:
            self.modelName = self.modelName + "(Retraining " + str(iter_retraining) + ")"

        self.resizedImagesPath = getPathResizedImage()
        self.objectsImagesPath = getPathObjectsImage()
        self.surfacesImagesPath = getPathSurfacesImage()
        self.threshold = float(os.environ.get('DIGITAL_MODEL_THRESHOLD_ANOMALY', 0.5))
        self.thresholdHigh = float(os.environ.get('DIGITAL_MODEL_HIGH_THRESHOLD_ANOMALY', 0.85))
        widthImage = int(os.environ.get('DIGITAL_MODEL_SIZE_IMAGES_WIDTH', 80))
        heightImage = int(os.environ.get('DIGITAL_MODEL_SIZE_IMAGES_HEIGHT', 45))
        self.sizeImage = [heightImage, widthImage]
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

    def save_evaluation_model(self, evaluation_dict):
        fullpath = getEvaluationModelPath()
        with open(fullpath, 'w') as file:
            file.write(json.dumps(evaluation_dict, indent=4))

    def get_evaluation_dict(self, model, X_tests, y_tests):
        evaulation_dict = get_evaluation_model(model, X_tests, y_tests)
        # Obtener las predicciones del modelo en el conjunto de test
        # y_pred = model.predict(X_test)
        # y_pred = np.argmax(y_pred, axis=1)
        return evaulation_dict

    def is_better_model(self, evaluation_dict, metric="f1_score"):
        try:
            actual_evaluation_dict = self.get_actual_evaluation_model()
            metric_obtained = float(evaluation_dict.get(metric, 0))
            metric_actual = float(actual_evaluation_dict.get(metric, 0))
            return metric_obtained >= metric_actual
        except Exception as e:
            logging.exception(e)
        return False

    def get_actual_evaluation_model(self):
        fullpath = getEvaluationModelPath()
        with open(fullpath, 'r') as file:
            evaluation_dict = json.load(file)
        return evaluation_dict

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

    @abstractmethod
    def __load_model__(self):
        pass

    @abstractmethod
    def get_evaluation_model(self, X, y):
        pass

    @abstractmethod
    def predict_sample(self, sample):
        pass

    @abstractmethod
    def save_model(self, model):
        pass

    @abstractmethod
    def get_actual_model_json(self):
        pass

    @abstractmethod
    def get_model_image_base64(self, model):
        pass

    @abstractmethod
    def __get_model_by_config__(self, retraining, retrainWeights, tunning, X_train, y_train, epochs, validation_split,
                                tuner=None):
        pass

    @abstractmethod
    def compile_model(self, model, optimizer="adam", metrics=["accuracy"], tunning=False, hp=None):
        pass

    @abstractmethod
    def create_model_layers(self, size_image, number_features):
        pass

    @abstractmethod
    def retrain_model(self, retraining=True, retrainWeights=True, tunning=False, model_by_best_epoch=False, random=None,
                      size_split=None, test_size=0.25, epochs=10, validation_split=0.2):
        pass