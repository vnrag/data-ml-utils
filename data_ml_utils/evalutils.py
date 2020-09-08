"""A library containing commonly used utils for machine learning evaluation
"""
import logging
from sklearn import metrics
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import json

class eval(object):
    logger=None
    algorithm_name=None
    timestamp=None
    train_data_name=None
    num_rows=None
    num_features=None
    cv_folds=None
    general_metrics= None
    xgb_metrics=None
    y_actual= None
    y_predicted= None
    y_predicted_prob= None
    
    def __init__(self, algorithm_name, timestamp, train_data_name, num_rows, num_features, cv_folds):
        self.logger= logging.getLogger("ML_Eval")
        self.logger.setLevel(logging.CRITICAL)
        self.algorithm_name= algorithm_name
        self.timestamp= timestamp
        self.train_data_name= train_data_name
        self.cv_folds= cv_folds
        self.general_metrics= general_eval()
        self.xgb_metrics= xgboost_eval()
    
    def prepare_eval(self, y_actual, y_predicted, y_predicted_prob=None):
        self.y_actual= y_actual
        self.y_predicted= y_predicted
        self.y_predicted_prob= y_predicted_prob
    
    def get_xgb_validation_metrics(self):
        return self.xgb_metrics.get_validation_metrics()
    
    def get_general_metrics(self):
        self.general_metrics.get_general_eval(self.y_actual, self.y_predicted)
    
    def get_xgb_eval_metrics(self, xgb_model):
        self.xgb_metrics.get_xgb_eval(xgb_model)
        
    def generate_xgb_eval_plots(self):
        self.xgb_metrics.generate_plots(self.y_actual, self.y_predicted_prob)

class general_eval(object):
    metrics= None
    
    def __init__(self):
        self.logger= logging.getLogger("General_Eval")
        self.logger.setLevel(logging.CRITICAL)
    
    def get_general_eval(self, y_actual, y_predicted):
        self.metrics= self.get_metrics_dict(y_actual, y_predicted)
    
    def confusion_matrix(self, y_actual, y_predicted):
       return pd.crosstab(y_actual, y_predicted, \
                        rownames=['Actual'], colnames=['Predicted'], \
                        margins=True, margins_name='Total')
    
    def accuracy(self, y_actual, y_predicted):
        return metrics.accuracy_score(y_actual, y_predicted)
    
    def f1(self, y_actual, y_predicted):
        return metrics.f1_score(y_actual, y_predicted)
    
    def precision(self, y_actual, y_predicted):
        return metrics.precision_score(y_actual, y_predicted)
    
    def recall(self, y_actual, y_predicted):
        return metrics.recall_score(y_actual, y_predicted)
    
    def roc_auc(self, y_actual, y_predicted):
        return metrics.roc_auc_score(y_actual, y_predicted)
    
    def get_metrics_dict(self, y_actual, y_predicted):
        eval_metrics={}
        eval_metrics['conf_matrix']= self.confusion_matrix(y_actual, y_predicted)
        eval_metrics['accuracy']= self.accuracy(y_actual, y_predicted)
        eval_metrics['f1']= self.f1(y_actual, y_predicted)
        eval_metrics['precision']= self.precision(y_actual, y_predicted)
        eval_metrics['recall']= self.recall(y_actual, y_predicted)
        eval_metrics['auc_roc']= self.roc_auc(y_actual, y_predicted)
        return eval_metrics

class xgboost_eval(object):
    model= None
    booster= None
    used_features= None
    metrics= None
    validation_metrics=['auc', 'rmse', 'mae', 'logloss', 'error', 'aucpr', 'map']

    def __init__(self):
        self.logger= logging.getLogger("XGB_Eval")
        self.logger.setLevel(logging.CRITICAL)
    
    def get_validation_metrics(self):
        return self.validation_metrics
    
    def get_xgb_eval(self, xgb_model):
        self.set_variables(xgb_model)
        self.metrics= self.get_metrics_dict()
        self.save_text_tree(xgb_model.get_booster())
    
    def set_variables(self, xgb_model):
        self.model= xgb_model
        self.booster= xgb_model.get_booster()
        self.used_features = xgb_model.get_booster().get_score().keys()
    
    def get_importance(self, xgb_model):
        importance_types = ['weight','gain','cover','total_gain','total_cover']
        importance={}
        for imp_type in importance_types:
            try:
                importance[imp_type] = xgb_model.get_score(importance_type= imp_type)
            except Exception as e:
                print(e)
        return importance

    def get_hist(self, xgb_model, used_features):
        hist={}
        for feature in used_features:
            try:
                hist[feature]= xgb_model.get_split_value_histogram(feature)
            except Exception as e:
                print(e)
        return hist
    
    def get_training_params(self, xgb_model):
        return xgb_model.get_params()
    
    def get_xgb_specific_params(self, xgb_model):
        return xgb_model.get_xgb_params()
    
    def get_validation_results(self, xgb_model):
        return xgb_model.evals_result()
    
    def get_model_config(self, booster):
        return json.loads(booster.save_config())
    
    def save_text_tree(self, booster):
        booster.dump_model('xgb_clf_dump.txt')
    
    def get_metrics_dict(self):
        metrics={}
        metrics['importance']= self.get_importance(self.booster)
        metrics['histogram']= self.get_hist(self.booster, self.used_features)
        metrics['training_params']= self.get_training_params(self.model)
        metrics['xgb_specific_params']= self.get_xgb_specific_params(self.model)
        metrics['validation_results']= self.get_validation_results(self.model)
        metrics['model_config']= self.get_model_config(self.booster)
        return metrics

    def generate_plots(self, y_actual, y_predicted_prob):
        self.get_importance_plots(self.model)
        self.get_tree_plot(self.model)
        self.get_roc_plot(y_actual, y_predicted_prob)
        self.get_pr_plot(y_actual, y_predicted_prob)
        
    
    def get_importance_plots(self, xgb_model):
        plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
        plt.savefig('imp1.png', bbox_inches='tight')
        plt.close()
        
        xgb.plot_importance(xgb_model)
        plt.savefig('imp2.png', bbox_inches='tight')
        plt.close()
    
    def get_tree_plot(self, xgb_model):
        fig= plt.figure(figsize=(50,10))
        xgb.plot_tree(xgb_model,num_trees=0)
        plt.savefig('tree.png', bbox_inches='tight')
        plt.close()
        
    def get_roc_plot(self, y_actual, y_predicted_prob):
        fpr, tpr, thresholds = metrics.roc_curve(y_actual, y_predicted_prob)
        plt.plot(fpr, tpr, marker='.', label='xgboost')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('roc.png', bbox_inches='tight')
        plt.close()
    
    def get_pr_plot(self, y_actual, y_predicted_prob):
        precision, recall, thresholds = metrics.precision_recall_curve(y_actual, y_predicted_prob)
        plt.plot(recall, precision, marker='.', label='xgboost')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig('pr.png', bbox_inches='tight')
        plt.close()
