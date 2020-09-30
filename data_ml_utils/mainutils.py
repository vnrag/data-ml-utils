"""A library containing shared functions between machine learning utils
"""
import logging
from data_utils.awsutils import S3Base, SSMBase
import os
from data_utils import generalutils as gu
import csv
import random

from .config import load_config
config = load_config()


class main_utils(object):
    logger=None
    logger_name='main_utils'
    atomic_metrics={}
    export_local= None
    export_s3= None
    s3_base= None
    ssm_base= None
    export_bucket= None
    local_folder= None
    s3_hash= None
    
    def __init__(self, project, dataset, use_case, setup, model_name, ts, num_rows, num_features, cv_folds, export_local= False, export_s3= False):
        self.logger= logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.CRITICAL)
        self.atomic_metrics['project']= project
        self.atomic_metrics['dataset']= dataset
        self.atomic_metrics['use_case']= use_case
        self.atomic_metrics['setup']= setup
        self.atomic_metrics['model']= model_name
        self.atomic_metrics['ts']= ts
        self.atomic_metrics['num_rows']= num_rows
        self.atomic_metrics['num_features']= num_features
        self.atomic_metrics['cv_folds']= cv_folds
        if export_local:
            self.export_local = export_local
            self.prepare_local_folder(model_name)
        if export_s3:
            self.export_s3 = export_s3
            self.s3_base= S3Base()
            self.ssm_base= SSMBase()
            self.export_bucket = self.ssm_base.get_ssm_parameter('MLBucketName', encoded = True)
        self.s3_hash = random.getrandbits(128)
        
    
    def prepare_local_folder(self, model_name):
        proj_folder= os.getcwd()
        self.local_folder= gu.get_target_path([proj_folder, model_name])
        if not os.path.exists(self.local_folder):
            os.makedirs(self.local_folder)
    
    def export_dict_as_text(self, data_dict , fname):
        file_path= gu.get_target_path([self.local_folder, fname], file_extension= 'csv')
        with open(file_path, 'w+') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data_dict.items():
                writer.writerow([key, value])
    
    def export_dict_as_one_row_text(self, data_dict, fname):
        csv_columns= data_dict.keys()
        file_path= gu.get_target_path([self.local_folder, fname], file_extension= 'csv')
        with open(file_path, 'w+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(data_dict)
    
    def export_df_as_text(self, df, fname):
        file_path= gu.get_target_path([self.local_folder, fname], file_extension='csv')
        df.to_csv(file_path, index=False, header=True)
    
    def get_metric_data_key(self, metric):
        keys= config['s3_structure']
        values=[metric, self.atomic_metrics['project'], self.atomic_metrics['dataset'], self.atomic_metrics['use_case'], self.atomic_metrics['setup']]
        datakeys= [k+'='+ v for k, v in zip(keys, values)]
        datakeys= ['ML_Analysis'] + datakeys
        return gu.get_target_path(datakeys)
    
    def export_metric_to_s3(self, df, key_name, file_name):
        file_name = file_name + "_%032x" % self.s3_hash
        datakey= self.get_metric_data_key(key_name)
        s3_uri= self.s3_base.create_s3_uri(self.export_bucket.decode(), datakey, file_name, FileType= 'parquet')
        self.s3_base.upload_parquet_with_wrangler(s3_uri, df)
