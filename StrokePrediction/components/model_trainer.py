import os
import sys

from StrokePrediction.exception.exception import StrokePredictionException
from StrokePrediction.logging.logger import logging

from StrokePrediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from StrokePrediction.entity.config_entity import ModelTrainerConfig

from StrokePrediction.utils.ml_utils.model.estimator import StrokePrediction
from StrokePrediction.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from StrokePrediction.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from urllib.parse import urlparse


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise StrokePredictionException(e, sys)


    # -------------------------
    # Train Model
    # -------------------------
    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
            },
            "Random Forest": {
                "n_estimators": [16, 32, 64, 128],
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05],
                "subsample": [0.7, 0.8, 0.9],
                "n_estimators": [32, 64, 128],
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [32, 64, 128],
            },
        }

        # -------------------------
        # Evaluate Models
        # -------------------------
        model_report, trained_models = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        # -------------------------
        # Select Best Model
        # -------------------------
        best_model_name = max(model_report, key=model_report.get)
        best_model = trained_models[best_model_name]
        best_model_score = model_report[best_model_name]

        logging.info(f"Best Model: {best_model_name}, F1 Score: {best_model_score}")

        # -------------------------
        # Train Metrics
        # -------------------------
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(
            y_true=y_train, y_pred=y_train_pred
        )



        # -------------------------
        # Test Metrics
        # -------------------------
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(
            y_true=y_test, y_pred=y_test_pred
        )


        # -------------------------
        # Load Preprocessor
        # -------------------------
        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        # -------------------------
        # Save FINAL MODEL (ONLY ONE)
        # -------------------------
        try:
                model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
                )
                os.makedirs(model_dir_path, exist_ok=True)

                StrokePrediction_Model = StrokePrediction(
                    preprocessor=preprocessor,
                    model=best_model,
                    #columns=None,  # FIXED (numpy array issue)
                )
        except Exception as e:
            logging.info("Unable to create model.pkl")
            raise StrokePredictionException(e,sys)
    

        logging.info("Saving final combined model...")

        save_object(
            self.model_trainer_config.trained_model_file_path,
            StrokePrediction_Model
        )


        save_object("final_model/model.pkl", best_model)

        # -------------------------
        # Return Artifact
        # -------------------------
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")

        return model_trainer_artifact

    # -------------------------
    # Entry Point
    # -------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise StrokePredictionException(e, sys)