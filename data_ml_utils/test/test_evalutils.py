"""Test Cases for evalutils library

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
test_data : TYPE
    path of folder for dummy testing dataset
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
import collections

from data_ml_utils.evalutils import general_eval, xgboost_eval
from sklearn.dummy import DummyClassifier
from sklearn import metrics

KEY_NAME = ['MLBucketName']
VALUE = ['Output']
MODEL_FOLDER = os.path.join(os.getcwd(), 'test_model')
now = datetime.now()
train_data = os.path.join(os.getcwd(), 'test', 'dummy', 'train_data.csv')
test_data = os.path.join(os.getcwd(), 'test', 'dummy', 'test_data.csv')


class TestGeneralEval(object):
    """Class for testing evalutils object
    """
    train = np.genfromtxt(train_data, delimiter=',')
    x_train = train[:, :-1]
    y_train = train[:, -1]

    test = np.genfromtxt(test_data, delimiter=',')
    x_test = test[:, :-1]
    y_test = test[:, -1]

    dummy_clf = DummyClassifier(strategy='uniform', random_state= 99)

    dummy_clf.fit(x_train, y_train)
    y_pred= dummy_clf.predict(x_test)
    y_proba= dummy_clf.predict_proba(x_test)

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

    def test_get_atomic_metrics(self, s3, ssm):
        """Tests get_atomic_metrics function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
               
        pairs = zip(geu.y_actual, self.y_test)
        assert len(geu.y_actual) == len(self.y_test)
        assert not any(x != y for x, y in pairs)
        
        pairs = zip(geu.y_predicted, self.y_pred)
        assert len(geu.y_predicted) == len(self.y_pred)
        assert not any(x != y for x, y in pairs)

        first_tuple_list = [tuple(lst) for lst in geu.y_predicted_prob]
        second_tuple_list = [tuple(lst) for lst in self.y_proba]
        pairs = zip(first_tuple_list, second_tuple_list)
        assert len(first_tuple_list) == len(second_tuple_list)
        assert not any(x != y for x, y in pairs)

        pairs = zip(geu.y_predicted_prob_one, self.y_proba[:,1])
        assert len(geu.y_predicted_prob_one) == len(self.y_proba[:,1])
        assert not any(x != y for x, y in pairs)
    
    def test_get_perf_metrics(self, s3, ssm):
        """Tests get_perf_metrics function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_perf_metrics()

        metrics= ['accuracy', 'f1', 'precision', 'recall', 'auc_roc']
        assert all(m in geu.atomic_metrics.keys() for m in metrics)
    
    def test_confusion_matrix(self, s3, ssm):
        """Tests confusion_matrix function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.confusion_matrix()
        assert not geu.conf_matrix.empty
        assert geu.conf_matrix['actual'].iloc[0] == 0
        assert geu.conf_matrix['actual'].iloc[1] == 0
        assert geu.conf_matrix['actual'].iloc[2] == 1
        assert geu.conf_matrix['actual'].iloc[3] == 1
        assert geu.conf_matrix['predicted'].iloc[0] == 0
        assert geu.conf_matrix['predicted'].iloc[1] == 1
        assert geu.conf_matrix['predicted'].iloc[2] == 0
        assert geu.conf_matrix['predicted'].iloc[3] == 1
        assert geu.conf_matrix['count'].iloc[0] == 1
        assert geu.conf_matrix['count'].iloc[1] == 2
        assert geu.conf_matrix['count'].iloc[2] == 1
        assert geu.conf_matrix['count'].iloc[3] == 2
        assert pd.Timestamp(geu.conf_matrix['ts'].unique()[0]).to_pydatetime() == now
        assert geu.conf_matrix['model'].unique()[0] == 'test_model'
    
    def test_accuracy(self, s3, ssm):
        """Tests accuracy function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.accuracy()
        assert 'accuracy' in geu.atomic_metrics.keys()
        assert geu.atomic_metrics['accuracy']== metrics.accuracy_score(self.y_test, self.y_pred)
    
    def test_f1(self, s3, ssm):
        """Tests f1 function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.f1()
        assert 'f1' in geu.atomic_metrics.keys()
        assert geu.atomic_metrics['f1']== metrics.f1_score(self.y_test, self.y_pred)
    
    def test_precision(self, s3, ssm):
        """Tests precision function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.precision()
        assert 'precision' in geu.atomic_metrics.keys()
        assert geu.atomic_metrics['precision']== metrics.precision_score(self.y_test, self.y_pred)

    def test_recall(self, s3, ssm):
        """Tests recall function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.recall()
        assert 'recall' in geu.atomic_metrics.keys()
        assert geu.atomic_metrics['recall']== metrics.recall_score(self.y_test, self.y_pred)
        
    def test_roc_auc(self, s3, ssm):
        """Tests roc_auc function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.roc_auc()
        assert 'auc_roc' in geu.atomic_metrics.keys()
        assert geu.atomic_metrics['auc_roc']== metrics.roc_auc_score(self.y_test, self.y_pred)

    def test_export_confusion_matrix_as_text(self, s3, ssm):
        """Tests export_confusion_matrix_as_text function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.confusion_matrix()
        geu.export_confusion_matrix_as_text()

        assert os.path.exists(os.path.join(MODEL_FOLDER, 'confusion_matrix.csv'))
        
        df_exported = pd.read_csv(os.path.join(MODEL_FOLDER, 'confusion_matrix.csv'))
        assert df_exported['actual'].iloc[0] == 0
        assert df_exported['actual'].iloc[1] == 0
        assert df_exported['actual'].iloc[2] == 1
        assert df_exported['actual'].iloc[3] == 1
        assert df_exported['predicted'].iloc[0] == 0
        assert df_exported['predicted'].iloc[1] == 1
        assert df_exported['predicted'].iloc[2] == 0
        assert df_exported['predicted'].iloc[3] == 1
        assert df_exported['count'].iloc[0] == 1
        assert df_exported['count'].iloc[1] == 2
        assert df_exported['count'].iloc[2] == 1
        assert df_exported['count'].iloc[3] == 2
        assert pd.Timestamp(df_exported['ts'].unique()[0]).to_pydatetime() == now
        assert df_exported['model'].unique()[0] == 'test_model'
    
    def test_export_confusion_matrix_to_s3(self, s3, ssm):
        """Tests export_atomic_metrics_to_s3 function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.confusion_matrix()
        geu.export_confusion_matrix_to_s3()

        bucket = geu.export_bucket.decode()
        key = geu.get_metric_data_key('confusion_matrix')
        file_name = 'confusion_matrix' + "_%032x" % geu.s3_hash
        full_key = os.path.join(key, f'{file_name}.parquet')
        s3_key = geu.s3_base.create_s3_uri(
            bucket, key, file_name, 'parquet')

        assert geu.s3_base.check_if_object_exists(
            bucket, full_key)

        df_exported = geu.s3_base.load_parquet_with_wrangler(s3_key)
        assert df_exported['actual'].iloc[0] == 0
        assert df_exported['actual'].iloc[1] == 0
        assert df_exported['actual'].iloc[2] == 1
        assert df_exported['actual'].iloc[3] == 1
        assert df_exported['predicted'].iloc[0] == 0
        assert df_exported['predicted'].iloc[1] == 1
        assert df_exported['predicted'].iloc[2] == 0
        assert df_exported['predicted'].iloc[3] == 1
        assert df_exported['count'].iloc[0] == 1
        assert df_exported['count'].iloc[1] == 2
        assert df_exported['count'].iloc[2] == 1
        assert df_exported['count'].iloc[3] == 2
        assert pd.Timestamp(df_exported['ts'].unique()[0]).to_pydatetime() == now
        assert df_exported['model'].unique()[0] == 'test_model'

    def test_export_atomic_metrics_as_text(self, s3, ssm):
        """Tests export_dict_as_one_row_text for atomic_metrics

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.export_dict_as_one_row_text(geu.atomic_metrics, 'atomic_metrics')

        assert os.path.exists(os.path.join(MODEL_FOLDER, 'atomic_metrics.csv'))

        df_exported = pd.read_csv(os.path.join(MODEL_FOLDER, 'atomic_metrics.csv'))
        cols= ['project', 'dataset', 'use_case', 'setup', 'model', 'ts', 'train_size',
       'validation_size', 'test_size', 'num_features', 'cv_folds', 'accuracy',
       'f1', 'precision', 'recall', 'auc_roc']
        assert all(c in df_exported.columns for c in cols)
        assert df_exported.iloc[0]['project']== geu.atomic_metrics['project']
        assert df_exported.iloc[0]['dataset']== geu.atomic_metrics['dataset']
        assert df_exported.iloc[0]['use_case']== geu.atomic_metrics['use_case']
        assert df_exported.iloc[0]['setup']== geu.atomic_metrics['setup']
        assert df_exported.iloc[0]['model']== geu.atomic_metrics['model']
        assert pd.Timestamp(df_exported.iloc[0]['ts']).to_pydatetime() == geu.atomic_metrics['ts']
        assert df_exported.iloc[0]['train_size']== geu.atomic_metrics['train_size']
        assert df_exported.iloc[0]['validation_size']== geu.atomic_metrics['validation_size']
        assert df_exported.iloc[0]['test_size']== geu.atomic_metrics['test_size']
        assert df_exported.iloc[0]['num_features']== geu.atomic_metrics['num_features']
        assert df_exported.iloc[0]['cv_folds']== geu.atomic_metrics['cv_folds']

    def test_export_atomic_metrics_to_s3(self, s3, ssm):
        """Tests export_atomic_metrics_to_s3 function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.export_atomic_metrics_to_s3()
        
        bucket = geu.export_bucket.decode()
        key = geu.get_metric_data_key('atomic_metrics')
        file_name = 'atomic_metrics' + "_%032x" % geu.s3_hash
        full_key = os.path.join(key, f'{file_name}.parquet')
        s3_key = geu.s3_base.create_s3_uri(
            bucket, key, file_name, 'parquet')

        assert geu.s3_base.check_if_object_exists(
            bucket, full_key)

        df_exported = geu.s3_base.load_parquet_with_wrangler(s3_key)
        cols= ['project', 'dataset', 'use_case', 'setup', 'model', 'ts', 'train_size',
       'validation_size', 'test_size', 'num_features', 'cv_folds', 'accuracy',
       'f1', 'precision', 'recall', 'auc_roc']
        assert all(c in df_exported.columns for c in cols)
        assert df_exported.iloc[0]['project']== geu.atomic_metrics['project']
        assert df_exported.iloc[0]['dataset']== geu.atomic_metrics['dataset']
        assert df_exported.iloc[0]['use_case']== geu.atomic_metrics['use_case']
        assert df_exported.iloc[0]['setup']== geu.atomic_metrics['setup']
        assert df_exported.iloc[0]['model']== geu.atomic_metrics['model']
        assert pd.Timestamp(df_exported.iloc[0]['ts']).to_pydatetime() == geu.atomic_metrics['ts']
        assert df_exported.iloc[0]['train_size']== geu.atomic_metrics['train_size']
        assert df_exported.iloc[0]['validation_size']== geu.atomic_metrics['validation_size']
        assert df_exported.iloc[0]['test_size']== geu.atomic_metrics['test_size']
        assert df_exported.iloc[0]['num_features']== geu.atomic_metrics['num_features']
        assert df_exported.iloc[0]['cv_folds']== geu.atomic_metrics['cv_folds']
        
    def test_get_roc_values(self, s3, ssm):
        """Tests get_roc_values function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_roc_values()
        fpr, tpr, _ = metrics.roc_curve(self.y_test, self.y_proba[:,1])
        assert 'roc' in geu.plots.keys()

        pairs = zip(geu.plots['roc']['fpr'].values, fpr)
        assert len(geu.plots['roc']['fpr'].values) == len(fpr)
        assert not any(x != y for x, y in pairs)

        pairs = zip(geu.plots['roc']['tpr'].values, tpr)
        assert len(geu.plots['roc']['tpr'].values) == len(tpr)
        assert not any(x != y for x, y in pairs)

    def test_get_roc_plot(self, s3, ssm):
        """Tests get_roc_plot function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_roc_values()
        geu.get_roc_plot()
        assert os.path.exists(os.path.join(MODEL_FOLDER, 'roc.png'))

    def test_get_pr_values(self, s3, ssm):
        """Tests get_pr_values function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_pr_values()
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, self.y_proba[:,1])        
        assert 'pr' in geu.plots.keys()

        pairs = zip(geu.plots['pr']['precision'].values, precision)
        assert len(geu.plots['pr']['precision'].values) == len(precision)
        assert not any(x != y for x, y in pairs)

        pairs = zip(geu.plots['pr']['recall'].values, recall)
        assert len(geu.plots['pr']['recall'].values) == len(recall)
        assert not any(x != y for x, y in pairs)

    def test_get_pr_plot(self, s3, ssm):
        """Tests get_pr_plot function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_pr_values()
        geu.get_pr_plot()
        assert os.path.exists(os.path.join(MODEL_FOLDER, 'pr.png'))

    def test_get_prob_values(self, s3, ssm):
        """Tests get_prob_values function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_prob_values()
        assert len(geu.plots['prob']) == 6
        probas= geu.plots['prob']['classification'].value_counts()
        assert probas['Negatives'] == 3
        assert probas['Positives'] == 3

    def test_get_prob_plot(self, s3, ssm):
        """Tests get_prob_plot function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_prob_values()
        geu.get_prob_plot()
        assert os.path.exists(os.path.join(MODEL_FOLDER, 'proba.png'))

    def test_export_roc_as_text(self, s3, ssm):
        """Tests export_roc_as_text function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_roc_values()
        geu.export_roc_as_text()
        fpr, tpr, _ = metrics.roc_curve(self.y_test, self.y_proba[:,1])

        assert os.path.exists(os.path.join(MODEL_FOLDER, 'roc_curve.csv'))

        df_exported = pd.read_csv(os.path.join(MODEL_FOLDER, 'roc_curve.csv'))
        pairs = zip(df_exported['fpr'].values, fpr)
        assert len(df_exported['fpr'].values) == len(fpr)
        assert not any(x != y for x, y in pairs)

        pairs = zip(df_exported['tpr'].values, tpr)
        assert len(df_exported['tpr'].values) == len(tpr)
        assert not any(x != y for x, y in pairs)

    def test_export_pr_as_text(self, s3, ssm):
        """Tests export_pr_as_text function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_pr_values()
        geu.export_pr_as_text()
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, self.y_proba[:,1])        

        assert os.path.exists(os.path.join(MODEL_FOLDER, 'pr_curve.csv'))

        df_exported = pd.read_csv(os.path.join(MODEL_FOLDER, 'pr_curve.csv'))

        pairs = zip(df_exported['precision'].values, precision)
        assert len(df_exported['precision'].values) == len(precision)
        assert not any(x != y for x, y in pairs)

        pairs = zip(df_exported['recall'].values, recall)
        assert len(df_exported['recall'].values) == len(recall)
        assert not any(x != y for x, y in pairs)

    def test_export_prob_plot_as_text(self, s3, ssm):
        """Tests export_prob_plot_as_text function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_prob_values()
        geu.export_prob_plot_as_text()

        assert os.path.exists(os.path.join(MODEL_FOLDER, 'class_probabilities.csv'))

        df_exported = pd.read_csv(os.path.join(MODEL_FOLDER, 'class_probabilities.csv'))

        assert len(df_exported) == 6
        probas= df_exported['classification'].value_counts()
        assert probas['Negatives'] == 3
        assert probas['Positives'] == 3

    def test_export_roc_to_s3(self, s3, ssm):
        """Tests export_roc_to_s3 function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_roc_values()
        geu.export_roc_to_s3()
        fpr, tpr, _ = metrics.roc_curve(self.y_test, self.y_proba[:,1])

        bucket = geu.export_bucket.decode()
        key = geu.get_metric_data_key('roc_curve')
        file_name = 'roc_curve' + "_%032x" % geu.s3_hash
        full_key = os.path.join(key, f'{file_name}.parquet')
        s3_key = geu.s3_base.create_s3_uri(
            bucket, key, file_name, 'parquet')

        assert geu.s3_base.check_if_object_exists(
            bucket, full_key)

        df_exported = geu.s3_base.load_parquet_with_wrangler(s3_key)
        pairs = zip(df_exported['fpr'].values, fpr)
        assert len(df_exported['fpr'].values) == len(fpr)
        assert not any(x != y for x, y in pairs)

        pairs = zip(df_exported['tpr'].values, tpr)
        assert len(df_exported['tpr'].values) == len(tpr)
        assert not any(x != y for x, y in pairs)

    def test_export_pr_to_s3(self, s3, ssm):
        """Tests export_pr_to_s3 function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_pr_values()
        geu.export_pr_to_s3()
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, self.y_proba[:,1])

        bucket = geu.export_bucket.decode()
        key = geu.get_metric_data_key('pr_curve')
        file_name = 'pr_curve' + "_%032x" % geu.s3_hash
        full_key = os.path.join(key, f'{file_name}.parquet')
        s3_key = geu.s3_base.create_s3_uri(
            bucket, key, file_name, 'parquet')

        assert geu.s3_base.check_if_object_exists(
            bucket, full_key)

        df_exported = geu.s3_base.load_parquet_with_wrangler(s3_key)
        pairs = zip(df_exported['precision'].values, precision)
        assert len(df_exported['precision'].values) == len(precision)
        assert not any(x != y for x, y in pairs)

        pairs = zip(df_exported['recall'].values, recall)
        assert len(df_exported['recall'].values) == len(recall)
        assert not any(x != y for x, y in pairs)

    def test_export_prob_plot_to_s3(self, s3, ssm):
        """Tests export_prob_plot_to_s3 function

        Parameters
        ----------
        s3 : object
            mocked s3 client object
        ssm : object
            mocked ssm client object
        """
        geu = general_eval(
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

        geu.get_atomic_metrics(self.y_test, self.y_pred, self.y_proba)
        geu.get_prob_values()
        geu.export_prob_plot_to_s3()
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, self.y_proba[:,1])

        bucket = geu.export_bucket.decode()
        key = geu.get_metric_data_key('class_probabilities')
        file_name = 'class_probabilities' + "_%032x" % geu.s3_hash
        full_key = os.path.join(key, f'{file_name}.parquet')
        s3_key = geu.s3_base.create_s3_uri(
            bucket, key, file_name, 'parquet')

        assert geu.s3_base.check_if_object_exists(
            bucket, full_key)

        df_exported = geu.s3_base.load_parquet_with_wrangler(s3_key)
        assert len(df_exported) == 6
        probas= df_exported['classification'].value_counts()
        assert probas['Negatives'] == 3
        assert probas['Positives'] == 3
        
        import IPython
        IPython.embed()