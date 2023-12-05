import logging
import numpy as np
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):



        pass

class MSE(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating_ MSE")
            mse =mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
    
class R2(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray): 
       
        try:
            logging.info("Entered the calculate_score method of the R2Score class")
            r2 = r2_score(y_true, y_pred)
            logging.info("The r2 score value is:{} ".format(r2))
            return r2
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the R2Score class. Exception message:  "
            )
            raise e
class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray): 
     
        try:
            logging.info("Entered the calculate_score method of the RMSE class")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("The root mean squared error value is: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the RMSE class. Exception message:  "
            )
            raise e
