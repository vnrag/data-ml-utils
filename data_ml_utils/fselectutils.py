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
        corr_mtx= x_pd.corr()
        corr_mtx.index.name ='feature'
        corr_mtx.reset_index(level=0, inplace=True)
        self.metrics['corr_matrix'] = corr_mtx
    
    def export_corr_matrix_as_text(self):
        df= self.metrics['corr_matrix']
        df.to_csv(r'feature_corr_matrix.csv', index = False, header=True)
    
    def save_dict_as_text(self, data_dict , fname):
        with open(f'{fname}.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data_dict.items():
                writer.writerow([key, value])
