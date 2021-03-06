"""A library containing commonly used utils for machine learning feature selection
"""
from data_ml_utils.mainutils import main_utils
from data_utils import generalutils as gu

class general_select(main_utils):
    logger_name='general_select'
    feature_metrics= {}

    def get_features_corr_matrix(self, matrix):
        x_pd= gu.create_data_frame(matrix)
        corr_mtx= x_pd.corr()
        ######################################################
        #should be removed when we have the feature names
        corr_mtx= corr_mtx.add_prefix('f')
        ######################################################
        corr_mtx= corr_mtx.melt(ignore_index= False)
        corr_mtx= corr_mtx.rename(columns={"variable": "feature_1"})
        corr_mtx.index.name ='feature'
        corr_mtx.reset_index(level=0, inplace=True)
        corr_mtx['model']= self.atomic_metrics['model']
        corr_mtx['ts']= self.atomic_metrics['ts']
        self.feature_metrics['corr_matrix'] = corr_mtx
    
    def export_metrics(self):
        if self.export_local:
            self.export_corr_matrix_as_text()
        if self.export_s3:
            self.export_corr_matrix_to_s3()
    
    def export_corr_matrix_as_text(self):
        df= self.feature_metrics['corr_matrix']
        
        self.export_df_as_text(df, 'feature_corr_matrix')
    
    def export_corr_matrix_to_s3(self):
        df= self.feature_metrics['corr_matrix']
        self.export_metric_to_s3(df, 'feature_corr_matrix', 'feature_corr_matrix')   
