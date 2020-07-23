import pickle
import numpy as np

import utils as ut
import config


class ModelPredictor:
    def __init__(self, input_factors: list):
        self.input_factors = input_factors
        self.model = pickle.load(open(config.trained_model, 'rb'))
        self.feature_importance = ModelPredictor.feature_ranker(self)

    def predictor(self) -> str:
        prediction = self.model.predict(self.input_factors)
        out = ut.prediction_translator(prediction, False)
        return out

    def prob_predictor(self) -> dict:
        """
        For predicting probability of the prediction
        :return: predicted prob
        """
        prediction_prob = self.model.predict_proba(self.input_factors)
        out = ut.prediction_translator(prediction_prob, True)
        return out

    def feature_ranker(self) -> np.ndarray:
        feature_importance = np.mean([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        factors = {
            "Age": feature_importance[0],
            "Sex": feature_importance[1],
            "Job": feature_importance[2],
            "Housing": feature_importance[3],
            "Savings account": feature_importance[4],
            "checking account": feature_importance[5],
            "Credit amount": feature_importance[6],
            "Duration": feature_importance[7],
            "Purpose": feature_importance[8]
        }
        return factors


