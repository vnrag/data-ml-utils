"""A library containing commonly used utils for machine learning feature selection
"""
import logging
import pandas as pd

class general_metrics(object):
    logger= None
    metrics= None
    
    def __init__(self):
        self.logger= logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)

    def get_features_corr_matrix(self, matrix):
        x_pd= pd.DataFrame(matrix)
        x_pd_corr = x_pd.corr()
        return x_pd_corr
