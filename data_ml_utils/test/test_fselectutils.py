"""Test Cases for fselectutils library

Attributes
----------
KEY_NAME : list
    list of mocked ssm parameter names
MODEL_FOLDER : TYPE
    path of folder for analysis results
now : datetime
    a static timestamp
VALUE : list
    list of mocked ssm parameter values
train_data : TYPE
    path of folder for dummy training dataset
"""
import os
import shutil
import pytest
import boto3
from unittest import mock
import pytest
from moto import mock_ssm, mock_s3
from datetime import datetime
import pandas as pd
import numpy as np

from data_ml_utils.fselectutils import general_select

KEY_NAME = ['MLBucketName']
VALUE = ['Output']
MODEL_FOLDER = os.path.join(os.getcwd(), 'test_model')
now = datetime.now()
train_data= os.path.join(os.getcwd(), 'test', 'dummy', 'train_data.csv')

class TestFSelectUtils(object):

    """Class for testing fselectutils object
    """
    
    @pytest.fixture(scope='function')
    def aws_credentials(self):
        """Mocked AWS Credentials for moto.
        """
        os.environ['AWS_ACCESS_KEY_ID'] = 'test'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'

    @pytest.fixture(scope='function')
    def s3(self, aws_credentials):
        """Mocking s3 for aws.
        
        Parameters
        ----------
        aws_credentials : dict
            Credentials for aws
        
        Yields
        ------
        [object] -- [boto3 client object.]
        """
        with mock_s3():
            s3 = boto3.client('s3', region_name='eu-central-1')
            buckets = self.create_buckets(s3)
            assert buckets['Buckets'][0]['Name'] == 'Output'
            yield s3

    @pytest.fixture(scope='function')
    def ssm(self, aws_credentials):
        """Puts parameters for s3.
        
        Parameters
        ----------
        aws_credentials : dict
            Credentials for aws
        
        Yields
        ------
        [object] -- [boto3 client object.]
        """
        with mock_ssm():
            ssm = boto3.client('ssm', region_name='eu-central-1')
            for i in range(0, len(KEY_NAME)):
                ssm.put_parameter(Name=KEY_NAME[i], Value=VALUE[i],
                                  Type='String')
            yield ssm

    def create_buckets(self, s3):
        """Created s3 bucket for given name.
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        
        Returns
        -------
        [dict] -- [Objects values from given buckets.]
        """
        s3.create_bucket(Bucket='Output', CreateBucketConfiguration={
                         'LocationConstraint': 'eu-central-1'})

        buckets = s3.list_buckets()
        return buckets

    def test_get_features_corr_matrix(self, s3, ssm):
        """Tests get_features_corr_matrix function
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        data= np.genfromtxt(train_data, delimiter=',')
        data= data[:,:-1]

        fsu = general_select(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=4,
            validation_size=0,
            test_size=4,
            num_features=4,
            cv_folds=0,
            export_local=True,
            export_s3=True
        )
        fsu.get_features_corr_matrix(data)

        df= fsu.feature_metrics['corr_matrix']

        assert len(df) == 16
        assert all(df['value'].iloc[i]==1 for i in [0,5,10,15])
        assert np.isclose(df['value'].iloc[7], 0.6862666932451417)
        assert np.isclose(df['value'].iloc[4], 0.9995588616980997)
        assert pd.Timestamp(df['ts'].unique()[0]).to_pydatetime() == now
        assert df['model'].unique()[0] == 'test_model'
    
    
    def test_export_corr_matrix_as_text(self, s3, ssm):
        """Tests export_corr_matrix_as_text function
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        fsu = general_select(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=4,
            validation_size=0,
            test_size=4,
            num_features=4,
            cv_folds=0,
            export_local=True,
            export_s3=True
        )
        
        data= np.genfromtxt(train_data, delimiter=',')
        data= data[:,:-1]
        fsu.get_features_corr_matrix(data)

        fsu.export_corr_matrix_as_text()
        df_exported = pd.read_csv(os.path.join(MODEL_FOLDER, 'feature_corr_matrix.csv'))

        assert len(df_exported) == 16
        assert all(df_exported['value'].iloc[i]==1 for i in [0,5,10,15])
        assert np.isclose(df_exported['value'].iloc[7], 0.6862666932451417)
        assert np.isclose(df_exported['value'].iloc[4], 0.9995588616980997)
        assert pd.Timestamp(df_exported['ts'].unique()[0]).to_pydatetime() == now
        assert df_exported['model'].unique()[0] == 'test_model'


    def test_export_corr_matrix_to_s3(self, s3, ssm):
        """Tests export_corr_matrix_to_s3 function
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        fsu = general_select(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=4,
            validation_size=0,
            test_size=4,
            num_features=4,
            cv_folds=0,
            export_local=True,
            export_s3=True
        )
        data= np.genfromtxt(train_data, delimiter=',')
        data= data[:,:-1]
        fsu.get_features_corr_matrix(data)

        fsu.export_corr_matrix_to_s3()

        bucket = fsu.export_bucket.decode()
        key = fsu.get_metric_data_key('feature_corr_matrix')
        file_name = 'feature_corr_matrix' + "_%032x" % fsu.s3_hash
        full_key = os.path.join(key, f'{file_name}.parquet')
        s3_key = fsu.s3_base.create_s3_uri(
            bucket, key, file_name, 'parquet')

        assert fsu.s3_base.check_if_object_exists(
            bucket, full_key)

        df_exported = fsu.s3_base.load_parquet_with_wrangler(s3_key)

        assert len(df_exported) == 16
        assert all(df_exported['value'].iloc[i]==1 for i in [0,5,10,15])
        assert np.isclose(df_exported['value'].iloc[7], 0.6862666932451417)
        assert np.isclose(df_exported['value'].iloc[4], 0.9995588616980997)
        assert pd.Timestamp(df_exported['ts'].unique()[0]).to_pydatetime() == now
        assert df_exported['model'].unique()[0] == 'test_model'
