"""A library of utils for machine learning Operations

Attributes
----------
config : dict
    Configuration elements
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
    """A parent class for all machine learning utils.
    It contains the basic shared attributes and functions.
    Attributes describe business purpose and technical details.
    Functions for handling of results like exporting them.

    Parameters
    ----------
    project : str
        Name of the project the analysis is done for
    dataset : str
        Name of the dataset used for the analysis
    use_case : str
        Name of the business scenario the analysis is done for
    setup : str
        Specification of the business scenario
    model_name : str
        Name of the machine learning model applied
    ts : datetime
        Timestamp when the analysis was done
    num_rows : int
        Number of rows in the dataset
    num_features : int
        Number of features of the dataset
    cv_folds : int
        Number of cross validation folds
    export_local : bool, optional
        Export evaluation metrics to local machine
    export_s3 : bool, optional
        Export evaluation metrics to s3


    Attributes
    ----------
    atomic_metrics : dict
        Metadata of the project, dataset and model
    export_bucket : str
        S3 bucket for exporting results
    export_local : bool
        Control exporting results to local machine
    export_s3 : bool
        Control exporting results to s3
    local_folder : str
        Path for local exports
    logger : Logger
        Logging object
    logger_name : str
        Name of the logger
    s3_base : S3Base
        Object for handling s3 operations
    s3_hash : int
        Random 128 bit hash. Added to parquet exports on s3. All files with the same hash belong to the same analysis
    ssm_base : SSMBase
        Object for handling ssm operations
    """

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
    
    def __init__(self, project, dataset, use_case, setup, model_name, ts, train_size, validation_size, test_size, num_features, cv_folds, export_local= False, export_s3= False):
        self.logger= logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.CRITICAL)
        self.atomic_metrics['project']= project
        self.atomic_metrics['dataset']= dataset
        self.atomic_metrics['use_case']= use_case
        self.atomic_metrics['setup']= setup
        self.atomic_metrics['model']= model_name
        self.atomic_metrics['ts']= ts
        self.atomic_metrics['train_size']= train_size
        self.atomic_metrics['validation_size']= validation_size
        self.atomic_metrics['test_size']= test_size
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
        """Creates a folder to export results on local machine.
        model_name parameter is used as the folder name.
        
        Parameters
        ----------
        model_name : str
            Name of the mahchine learning model
        """
        proj_folder= os.getcwd()
        self.local_folder= gu.get_target_path([proj_folder, model_name])
        if not os.path.exists(self.local_folder):
            os.makedirs(self.local_folder)
    
    def export_dict_as_text(self, data_dict , fname):
        """Writes a dictionary to a text file.
        File is exported to a folder holding the machine learning model name
        based on the model_name parameter
        
        Parameters
        ----------
        data_dict : dict
            Dictinary of data to be exported
        fname : str
            Name of the file
        """
        file_path= gu.get_target_path([self.local_folder, fname], file_extension= 'csv')
        with open(file_path, 'w+') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data_dict.items():
                writer.writerow([key, value])
    
    def export_dict_as_one_row_text(self, data_dict, fname):
        """Writes a dictionary to a text file after pivoting it's data
        into one row. File is exported to a folder holding the machine
        learning model name based on the model_name parameter
        
        Parameters
        ----------
        data_dict : dict
            Dictinary of data to be exported
        fname : str
            Name of the file
        """
        csv_columns= data_dict.keys()
        file_path= gu.get_target_path([self.local_folder, fname], file_extension= 'csv')
        with open(file_path, 'w+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(data_dict)
    
    def export_df_as_text(self, df, fname):
        """Writes a dataframe to a text file.
        File is exported to a folder holding the machine learning model name
        based on the model_name parameter
        
        Parameters
        ----------
        df : Pandas DataFrame
            DataFrame of data to be exported
        fname : str
            Name of file
        """
        file_path= gu.get_target_path([self.local_folder, fname], file_extension='csv')
        df.to_csv(file_path, index=False, header=True)
    
    def get_metric_data_key(self, metric):
        """Creates an s3 key for exporting results of a metric.
        Uses s3_structure from configuration dictionary for partitions structure.
        Uses metadata from atomic_metrics attribute to fill the partition values.

        Parameters
        ----------
        metric : str
            Name of metric. Used to 
        
        Returns
        -------
        str
            S3 key for exporting results of the metric
        """
        keys= config['s3_structure']
        values=[metric, self.atomic_metrics['project'], self.atomic_metrics['dataset'], self.atomic_metrics['use_case'], self.atomic_metrics['setup']]
        datakeys= [k+'='+ v for k, v in zip(keys, values)]
        datakeys= ['ML_Analysis'] + datakeys
        return gu.get_target_path(datakeys)
    
    def export_metric_to_s3(self, df, key_name, file_name):
        """Writes a dataframe to s3 in parquet format.
        
        Parameters
        ----------
        df : Pandas DataFrame
            DataFrame of data to be exported
        key_name : str
            Name of s3 partition
        file_name : str
            Name of file
        """
        file_name = file_name + "_%032x" % self.s3_hash
        datakey= self.get_metric_data_key(key_name)
        s3_uri= self.s3_base.create_s3_uri(self.export_bucket.decode(), datakey, file_name, FileType= 'parquet')
        self.s3_base.upload_parquet_with_wrangler(s3_uri, df)
