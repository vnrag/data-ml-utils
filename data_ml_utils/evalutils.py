"""A library containing commonly used utils for machine learning evaluation
"""
import logging
from sklearn import metrics
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import csv
from data_utils.awsutils import S3Base, SSMBase
from data_utils import generalutils as gu
import numpy as np

class general_eval(object):
    eval_type="general_eval"
    logger=None
    atomic_metrics={}
    conf_matrix= None
    model_metrics={}
    y_actual= None
    y_predicted= None
    y_predicted_prob= None
    export_local= None
    export_s3= None
    s3_base= None
    ssm_base= None
    export_bucket= None
    
    def __init__(self, project, dataset, use_case, setup, model_name, ts, num_rows, num_features, cv_folds, export_local= False, export_s3= False):
        self.logger= logging.getLogger(self.eval_type)
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
        self.export_local = export_local
        if export_s3:
            self.export_s3 = export_s3
            self.s3_base= S3Base()
            self.ssm_base= SSMBase()
            self.export_bucket = self.ssm_base.get_ssm_parameter('MLBucketName', encoded = True)

    def get_atomic_metrics(self, y_actual, y_predicted, y_predicted_prob):
        self.y_actual= y_actual
        self.y_predicted= y_predicted
        self.y_predicted_prob= y_predicted_prob
        self.y_predicted_prob_one= y_predicted_prob[:,1]
        self.get_perf_metrics()
        self.conf_matrix= self.confusion_matrix()
    
    def get_perf_metrics(self):
        self.atomic_metrics['accuracy']= self.accuracy()
        self.atomic_metrics['f1']= self.f1()
        self.atomic_metrics['precision']= self.precision()
        self.atomic_metrics['recall']= self.recall()
        self.atomic_metrics['auc_roc']= self.roc_auc()
    
    def confusion_matrix(self):
        conf_df= pd.DataFrame({'actual':self.y_actual,
                          'predicted':self.y_predicted})
        conf_df= conf_df.groupby(['actual','predicted'], as_index=False).size()
        conf_df.rename(columns={'size': 'count'})
        conf_df['model']= self.atomic_metrics['model']
        conf_df['ts']= self.atomic_metrics['ts']
        return conf_df
    
    def accuracy(self):
        return metrics.accuracy_score(self.y_actual, self.y_predicted)
    
    def f1(self):
        return metrics.f1_score(self.y_actual, self.y_predicted)
    
    def precision(self):
        return metrics.precision_score(self.y_actual, self.y_predicted)
    
    def recall(self):
        return metrics.recall_score(self.y_actual, self.y_predicted)
    
    def roc_auc(self):
        return metrics.roc_auc_score(self.y_actual, self.y_predicted)
    
    def export_atomic_metrics(self):
        if self.export_local:
            self.save_dict_as_one_row_text(self.atomic_metrics, 'atomic_metrics')
            self.export_confusion_matrix_as_text()
        if self.export_s3:
            self.export_atomic_metrics_to_s3()
            self.export_confusion_matrix_to_s3()
    
    def export_confusion_matrix_as_text(self):
        df= self.conf_matrix
        df.to_csv(r'confusion_matrix.csv', index = False, header=True)
    
    def export_atomic_metrics_to_s3(self):
        df= gu.normalize_json(self.atomic_metrics)
        self.export_metric_to_s3(df, 'atomic_metrics', 'atomic_metrics')
    
    def export_confusion_matrix_to_s3(self):
        df= self.conf_matrix
        self.export_metric_to_s3(df, 'confusion_matrix', 'confusion_matrix')
    
    def save_dict_as_text(self, data_dict , fname):
        with open(f'{fname}.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data_dict.items():
                writer.writerow([key, value])
    
    def get_metric_data_key(self, metric):
        keys=['type', 'project', 'dataset', 'use_case', 'setup']
        values=[metric, self.atomic_metrics['project'], self.atomic_metrics['dataset'], self.atomic_metrics['use_case'], self.atomic_metrics['setup']]
        datakeys= [k+'='+ v for k, v in zip(keys, values)]
        datakeys= ["glue"] + datakeys
        return gu.get_target_path([f"glue",datakeys])
    
    def save_dict_as_one_row_text(self, data_dict, fname):
        csv_columns= data_dict.keys()
        with open(f'{fname}.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(data_dict)
    
    def export_metric_to_s3(self, df, key_name, file_name):
        import IPython
        IPython.embed()
        datakey= self.get_metric_data_key(key_name)
        s3_uri= self.s3_base.create_s3_uri(self.export_bucket, datakey, file_name, FileType= "parquet")
        self.s3_base.upload_parquet_with_wrangler(s3_uri, df)

class xgboost_eval(general_eval):
    eval_type="xgb_eval"
    model= None
    booster= None
    used_features= None
    plots={}
    validation_metrics=['auc', 'rmse', 'mae', 'logloss', 'error', 'aucpr', 'map']
    
    def get_validation_metrics(self):
        return self.validation_metrics
    
    def get_eval(self, xgb_model):
        self.set_variables(xgb_model)
        self.get_model_metrics()
    
    def set_variables(self, xgb_model):
        self.model= xgb_model
        self.booster= xgb_model.get_booster()
        self.used_features = xgb_model.get_booster().get_score().keys()
    
    def get_model_metrics(self):
        self.model_metrics['importance']= self.get_importance()
        self.model_metrics['histogram']= self.get_hist()
        self.model_metrics['training_params']= self.get_training_params()
        self.model_metrics['xgb_specific_params']= self.get_xgb_specific_params()
        self.model_metrics['validation_results']= self.get_validation_results()
        self.model_metrics['model_config']= self.get_model_config()
    
    def get_importance(self):
        importance_types = ['weight','gain','cover','total_gain','total_cover']
        importance={}
        for imp_type in importance_types:
            try:
                importance[imp_type] = self.booster.get_score(importance_type= imp_type)
            except Exception as e:
                print(e)
        importance_df= pd.DataFrame(importance)
        importance_df.index.name ='feature'
        importance_df.reset_index(level=0, inplace=True)
        importance_df['model']= self.atomic_metrics['model']
        importance_df['ts']= self.atomic_metrics['ts']
        return importance_df

    def get_hist(self):
        hist={}
        for feature in self.used_features:
            try:
                df= self.booster.get_split_value_histogram(feature)
                df.index.name= 'split_index'
                df.reset_index(level=0, inplace=True)
                df['feature']= feature
                hist[feature]= df
            except Exception as e:
                print(f'Feature {feature}:')
                print(e)
        hist_df = pd.concat(hist.values(), ignore_index=True)
        # rearrange columns
        cols = hist_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        hist_df= hist_df[cols]
        hist_df['mode']= self.atomic_metrics['model']
        hist_df['ts']= self.atomic_metrics['ts']
        return hist_df
    
    def get_training_params(self):
        return self.model.get_params()
    
    def get_xgb_specific_params(self):
        return self.model.get_xgb_params()
    
    def get_validation_results(self):
        val_results= self.model.evals_result()
        train= pd.DataFrame(val_results['validation_0'])
        train= train.add_prefix('train_')
        test= pd.DataFrame(val_results['validation_1'])
        test= test.add_prefix('test_')
        train_test= pd.concat([train, test],axis=1)
        train_test.index.name='epoch'
        train_test.reset_index(level=0, inplace= True)
        train_test['model']= self.atomic_metrics['model']
        train_test['ts']= self.atomic_metrics['ts']
        return train_test
    
    def get_model_config(self):
        return json.loads(self.booster.save_config())

    def export_model_metrics(self):
        combined_metrics = {**self.atomic_metrics, **self.model_metrics['xgb_specific_params']}
        if self.export_local:
            self.save_dict_as_one_row_text(combined_metrics, 'xgboost_metrics')
            self.export_histogram_as_text()
            self.save_dict_as_text(self.model_metrics['model_config'], 'model_config')
        if self.export_s3:
            self.export_model_metrics_to_s3(combined_metrics)
            self.export_histogram_to_s3()
            self.export_model_config_to_s3()
    
    def export_histogram_as_text(self):
        df= self.model_metrics['histogram']
        df.to_csv(r'feature_histogram.csv', index=False, header=True)
    
    def export_model_metrics_to_s3(self, combined_metrics):
        df= gu.normalize_json(combined_metrics)
        self.export_metric_to_s3(df, 'xgboost_metrics', 'xgboost_metrics')
    
    def export_histogram_to_s3(self):
        df= self.model_metrics['histogram']
        self.export_metric_to_s3(df, 'feature_histogram', 'feature_histogram')
    
    def export_model_config_to_s3(self):
        df= gu.normalize_json(self.model_metrics['model_config'])
        self.export_metric_to_s3(df, 'model_config', 'model_config')

    def get_plots(self):
        self.get_validation_metrics_plots()
        self.get_importance_plots()
        self.get_tree_plot()
        self.get_roc_values()
        self.get_roc_plot()
        self.get_pr_values()
        self.get_pr_plot()
        self.get_prob_values()
        self.get_prob_plot()
        
    def get_importance_plots(self):
        plt.rcParams.update(plt.rcParamsDefault)
        plt.bar(range(len(self.model.feature_importances_)), self.model.feature_importances_)
        plt.savefig('imp1.png', bbox_inches='tight')
        plt.close()
        
        plt.rcParams.update(plt.rcParamsDefault)
        xgb.plot_importance(self.model)
        plt.savefig('imp2.png', bbox_inches='tight')
        plt.close()
    
    def get_tree_plot(self):
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['figure.figsize'] = [50, 10]
        xgb.plot_tree(self.model,num_trees=0)
        plt.savefig('tree.png', bbox_inches='tight')
        plt.close()
        
    def get_roc_plot(self):
        fpr= self.plots['roc']['fpr']
        tpr= self.plots['roc']['tpr']
        plt.rcParams.update(plt.rcParamsDefault)
        plt.plot(fpr, tpr, marker='.', label='xgboost')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('roc.png', bbox_inches='tight')
        plt.close()
    
    def get_roc_values(self):
        fpr, tpr, _ = metrics.roc_curve(self.y_actual, self.y_predicted_prob_one)
        roc_df= pd.DataFrame({'fpr':fpr, 'tpr': tpr})
        roc_df.index.name='index'
        roc_df.reset_index(level=0, inplace= True)
        roc_df['model']= self.atomic_metrics['model']
        roc_df['ts']= self.atomic_metrics['ts']
        self.plots['roc']= roc_df
    
    def get_pr_plot(self):
        plt.rcParams.update(plt.rcParamsDefault)
        precision= self.plots['pr']['precision']
        recall= self.plots['pr']['recall']
        plt.plot(recall, precision, marker='.', label='xgboost')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig('pr.png', bbox_inches='tight')
        plt.close()
    
    def get_pr_values(self):
        precision, recall, _ = metrics.precision_recall_curve(self.y_actual, self.y_predicted_prob_one)
        pr_df= pd.DataFrame({'precision':precision, 'recall': recall})
        pr_df.index.name='index'
        pr_df.reset_index(level=0, inplace= True)
        pr_df['model']= self.atomic_metrics['model']
        pr_df['ts']= self.atomic_metrics['ts']
        self.plots['pr']= pr_df
    
    def get_validation_metrics_plots(self):
        for metric in self.validation_metrics:
            df= self.model_metrics['validation_results'][[f'test_{metric}',f'train_{metric}']]
            df.plot()
            plt.ylabel(metric)
            plt.xlabel('epochs')
            plt.title(f'XGBoost {metric}')
            plt.savefig(f'val_{metric}.png', bbox_inches='tight')
            plt.close()
    
    def get_prob_plot(self):
        df= self.plots['prob']
        neg= df[df.classification=='Negatives']['prob_class_1']
        pos= df[df.classification=='Positives']['prob_class_1']
        plt.hist(neg, bins=100, label='Negatives')
        plt.hist(pos, bins=100, label='Positives', alpha=0.7, color='r')
        plt.xlabel('Probability of being Positive Class')
        plt.ylabel('Number of records in each bucket')
        plt.legend()
        plt.tick_params(axis='both', pad=5)
        plt.savefig(f'proba.png', bbox_inches='tight')
        plt.close()
    
    def get_prob_values(self):
        cols = ['predicted','actual','prob_class_1']
        data= np.column_stack([self.y_predicted, self.y_actual, self.y_predicted_prob_one])
        prob_df = pd.DataFrame(data= data, columns = cols)
        prob_df['classification'] = np.where(prob_df['predicted']==prob_df['actual'],'Positives','Negatives')
        prob_df.index.name='index'
        prob_df.reset_index(level=0, inplace=True)
        prob_df['model']= self.atomic_metrics['model']
        prob_df['ts']= self.atomic_metrics['ts']
        self.plots['prob']= prob_df
    
    def export_plots(self):
        if self.export_local:
            self.export_validation_as_text()
            self.export_importance_as_text()
            self.export_tree_as_text()
            self.export_roc_as_text()
            self.export_pr_as_text()
            self.export_prob_plot_as_text()
        if self.export_s3:
            self.export_validation_to_s3()
            self.export_importance_to_s3()
            # self.export_tree_to_s3()
            self.export_roc_to_s3()
            self.export_pr_to_s3()
            self.export_prob_plot_to_s3()
    
    def export_importance_as_text(self):
        df= self.model_metrics['importance']
        df.to_csv(r'imp.csv', index=False, header=True)
    
    def export_tree_as_text(self):
        self.booster.dump_model('tree.csv')
    
    def export_roc_as_text(self):
        df= self.plots['roc']
        df.to_csv(r'roc.csv', index = False, header=True)
    
    def export_pr_as_text(self):
        df= self.plots['pr']
        df.to_csv(r'pr.csv', index = False, header=True)
    
    def export_validation_as_text(self):
        df= self.model_metrics['validation_results']
        df.to_csv(r'validation_results.csv', index = False, header=True)
    
    def export_prob_plot_as_text(self):
        df = self.plots['prob']
        df.to_csv(r'proba.csv', index=False, header= True)

    def export_importance_to_s3(self):
        df= self.model_metrics['importance']
        self.export_metric_to_s3(df, 'feature_importance', 'feature_importance')
    
    # def export_tree_to_s3(self):
    #     self.booster.dump_model('tree.csv')
    
    def export_roc_to_s3(self):
        df= self.plots['roc']
        self.export_metric_to_s3(df, 'roc_curve', 'roc_curve')
    
    def export_pr_to_s3(self):
        df= self.plots['pr']
        self.export_metric_to_s3(df, 'pr_curve', 'pr_curve')
    
    def export_validation_to_s3(self):
        df= self.model_metrics['validation_results']
        self.export_metric_to_s3(df, 'validation_results', 'validation_results')
    
    def export_prob_plot_to_s3(self):
        df = self.plots['prob']
        self.export_metric_to_s3(df, 'class_probabilities', 'class_probabilities')
