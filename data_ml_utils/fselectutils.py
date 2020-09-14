"""A library containing commonly used utils for machine learning feature selection
"""
import logging
import pandas as pd
import csv

class general_metrics(object):
    logger= None
    metrics= {}
    
    def __init__(self):
        self.logger= logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)

    def get_features_corr_matrix(self, matrix):
        x_pd= pd.DataFrame(matrix)
        self.metrics['corr_matrix'] = x_pd.corr()
    
    def export_general_metrics_as_text(self):
        self.save_dict_as_text(self.metrics, 'f_selection')
    
    def save_dict_as_text(self, data_dict , fname):
        with open(f'{fname}.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data_dict.items():
                writer.writerow([key, value])
