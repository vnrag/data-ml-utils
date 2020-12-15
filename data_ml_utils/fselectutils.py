"""A library of utils for machine learning feature selection
"""
from data_ml_utils.mainutils import main_utils
from data_utils import generalutils as gu
import pandas as pd

class general_select(main_utils):
    """Class for general feature selection utilities
    
    Attributes
    ----------
        feature_metrics : dict
            Dictionary containing calculated metrics
        logger_name : str
            Name of Logger
    """
    logger_name='general_select'
    feature_metrics= {}

    def get_features_corr_matrix(self, matrix):
        """Calculates a matrix of correlation between
        pairs of features
        
        Args:
            matrix : DataFrame
                Pandas dataframe containing features
        """
        x_pd= gu.create_data_frame(matrix)
        corr_mtx= x_pd.corr()
        ######################################################
        #should be removed when we have the feature names
        # corr_mtx= corr_mtx.add_prefix('f')
        ######################################################
        corr_mtx.index.name ='feature'
        corr_mtx.reset_index(level=0, inplace=True)
        corr_mtx= pd.melt(corr_mtx, id_vars =['feature'], var_name ='feature_1')
        corr_mtx['model']= self.atomic_metrics['model']
        corr_mtx['ts']= self.atomic_metrics['ts']
        self.feature_metrics['corr_matrix'] = corr_mtx
    
    def export_metrics(self):
        """Exports each feature selection metric to a separate file.
            If export_local flag is set to True, metrics are saved to
            text files on the local machine. If export_s3 flag is set
            to True, metrics are saved to parquet files on s3.
        """
        if self.export_local:
            self.export_corr_matrix_as_text()
        if self.export_s3:
            self.export_corr_matrix_to_s3()
    
    def export_corr_matrix_as_text(self):
        """Export correlation matrix metric to local disk in csv format
        """
        df= self.feature_metrics['corr_matrix']
        self.export_df_as_text(df, 'feature_corr_matrix')
    
    def export_corr_matrix_to_s3(self):
        """Export correlation matrix metric to s3 in parquet format
        """
        df= self.feature_metrics['corr_matrix']
        self.export_metric_to_s3(df, 'feature_corr_matrix', 'feature_corr_matrix')   
