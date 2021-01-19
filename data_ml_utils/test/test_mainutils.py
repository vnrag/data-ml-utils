"""Test Cases for mainutils library

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

from data_ml_utils.mainutils import main_utils

KEY_NAME = ['MLBucketName']
VALUE = ['Output']
MODEL_FOLDER = os.path.join(os.getcwd(), 'test_model')
now = datetime.now()


class TestMainUtils(object):

    """Class for testing mainutils object
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
    
    def test_init(self, s3, ssm):
        """Tests creation of mainutils object
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """

        mu = main_utils(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=100,
            validation_size=50,
            test_size=50,
            num_features=20,
            cv_folds=5,
            export_local=True,
            export_s3=True
        )

        assert mu.atomic_metrics['project'] == 'test_proj'
        assert mu.atomic_metrics['dataset'] == 'test_ds'
        assert mu.atomic_metrics['use_case'] == 'test_uc'
        assert mu.atomic_metrics['setup'] == 'test_stp'
        assert mu.atomic_metrics['model'] == 'test_model'
        assert mu.atomic_metrics['ts'] == now
        assert mu.atomic_metrics['train_size'] == 100
        assert mu.atomic_metrics['validation_size'] == 50
        assert mu.atomic_metrics['test_size'] == 50
        assert mu.atomic_metrics['num_features'] == 20
        assert mu.atomic_metrics['cv_folds'] == 5
        assert mu.export_local
        assert mu.export_s3
        assert mu.export_bucket.decode() == 'Output'
        assert mu.s3_hash
        assert os.path.exists(MODEL_FOLDER)

    def test_export_dict_as_text(self, s3, ssm):
        """Tests export_dict_as_text function
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        mu = main_utils(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=100,
            validation_size=50,
            test_size=50,
            num_features=20,
            cv_folds=5,
            export_local=True,
            export_s3=True
        )

        data = {
            'Name': 'Value',
            'project': 'test_project',
            'dataset': 'test_ds',
            'use_case': 'test_uc',
            'setup': 'test_stp',
            'model_name': 'test_model'
        }

        mu.export_dict_as_text(data, 'test_dict')
        df = pd.read_csv(os.path.join(MODEL_FOLDER, 'test_dict.csv'))
        len(df) == 5
        df.columns.to_list() == ['Name', 'Value']
        df.iloc[0]['Name'] == 'project'
        df.iloc[1]['Value'] == 'test_ds'

    def test_export_dict_as_one_row_text(self, s3, ssm):
        """Tests export_dict_as_one_row_text function
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        mu = main_utils(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=100,
            validation_size=50,
            test_size=50,
            num_features=20,
            cv_folds=5,
            export_local=True,
            export_s3=True
        )

        data = {
            'project': 'test_project',
            'dataset': 'test_ds',
            'use_case': 'test_uc',
            'setup': 'test_stp',
            'model_name': 'test_model'
        }

        mu.export_dict_as_one_row_text(data, 'test_dict_pivotted')
        df = pd.read_csv(os.path.join(MODEL_FOLDER, 'test_dict_pivotted.csv'))
        len(df) == 1
        df.columns.to_list() == ['project', 'dataset',
                                 'use_case', 'setup', 'model_name']
        df.iloc[0]['setup'] == 'test_stp'
        df.iloc[0]['model_name'] == 'test_model'

    def test_export_df_as_text(self, s3, ssm):
        """Tests export_df_as_text function
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        mu = main_utils(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=100,
            validation_size=50,
            test_size=50,
            num_features=20,
            cv_folds=5,
            export_local=True,
            export_s3=True
        )
        data = {
            'Name': ['project', 'dataset', 'use_case', 'setup', 'model_name'],
            'Value': ['test_project', 'test_ds', 'test_uc', 'test_stp', 'test_model']
        }
        df = pd.DataFrame.from_dict(data)
        mu.export_df_as_text(df, 'test_df')

        df_exported = pd.read_csv(os.path.join(MODEL_FOLDER, 'test_df.csv'))
        len(df_exported) == 5
        df_exported.columns.to_list() == ['Name', 'Value']
        df_exported.iloc[0]['Name'] == 'project'
        df_exported.iloc[1]['Value'] == 'test_ds'

    def test_get_metric_data_key(self, s3, ssm):
        """Tests get_metric_data_key function
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        mu = main_utils(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=100,
            validation_size=50,
            test_size=50,
            num_features=20,
            cv_folds=5,
            export_local=True,
            export_s3=True
        )

        key = mu.get_metric_data_key('test_metric')
        paritions = key.split('/')
        assert paritions[0] == 'ML_Analysis'
        assert paritions[1] == 'partition_type=test_metric'
        assert paritions[2] == f"partition_project={mu.atomic_metrics['project']}"
        assert paritions[3] == f"partition_dataset={mu.atomic_metrics['dataset']}"
        assert paritions[4] == f"partition_use_case={mu.atomic_metrics['use_case']}"
        assert paritions[5] == f"partition_setup={mu.atomic_metrics['setup']}"

    def test_export_metric_to_s3(self, s3, ssm):
        """Tests export_metric_to_s3 function
        
        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        mu = main_utils(
            project='test_proj',
            dataset='test_ds',
            use_case='test_uc',
            setup='test_stp',
            model_name='test_model',
            ts=now,
            train_size=100,
            validation_size=50,
            test_size=50,
            num_features=20,
            cv_folds=5,
            export_local=True,
            export_s3=True
        )
        data = {
            'Name': ['project', 'dataset', 'use_case', 'setup', 'model_name'],
            'Value': ['test_project', 'test_ds', 'test_uc', 'test_stp', 'test_model']
        }

        df = pd.DataFrame.from_dict(data)
        mu.export_metric_to_s3(df, 'test_metric', 'test_file')

        bucket = mu.export_bucket.decode()
        key = mu.get_metric_data_key('test_metric')
        file_name = 'test_file' + "_%032x" % mu.s3_hash
        full_key = os.path.join(key, f'{file_name}.parquet')
        s3_key = mu.s3_base.create_s3_uri(
            bucket, key, file_name, 'parquet')

        assert mu.s3_base.check_if_object_exists(
            bucket, full_key)

        df_exported = mu.s3_base.load_parquet_with_wrangler(s3_key)

        assert len(df) == 5
        assert df_exported.columns.to_list() == ['Name', 'Value']
        assert df_exported.iloc[0]['Name'] == 'project'
        assert df_exported.iloc[1]['Value'] == 'test_ds'
