"""A library containing commonly used utils for machine learning evaluation
"""
import logging
from sklearn import metrics
import pandas as pd
import xgboost as xgb

class general_metrics(object):
    logger= None
    y_actual= None
    y_predicted= None
    metrics= None
    
    def __init__(self, y_actual, y_predicted):
        self.logger= logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)
        self.y_actual= y_actual
        self.y_predicted= y_predicted
        self.metrics= self.get_metrics_dict()
    
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
    
    def get_metrics_dict(self):
        eval_metrics={}
        eval_metrics['conf_matrix']= self.confusion_matrix(self.y_actual, self.y_predicted)
        eval_metrics['accuracy']= self.accuracy(self.y_actual, self.y_predicted)
        eval_metrics['f1']= self.f1(self.y_actual, self.y_predicted)
        eval_metrics['precision']= self.precision(self.y_actual, self.y_predicted)
        eval_metrics['recall']= self.recall(self.y_actual, self.y_predicted)
        eval_metrics['auc_roc']= self.roc_auc(self.y_actual, self.y_predicted)
        return eval_metrics

class xgboost_eval(object):
    logger= None
    model= None
    booster= None
    used_features= None
    metrics= None
    
    def __init__(self, xgb_model):
        self.logger= logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)
        self.model= xgb_model
        self.booster= xgb_model.get_booster()
        self.used_features = xgb_model.get_booster().get_score().keys()
        self.metrics= self.get_metrics_dict()
    
    def get_importance(self, model):
        importance_types = ['weight','gain','cover','total_gain','total_cover']
        importance={}
        for imp_type in importance_types:
            try:
                importance[imp_type] = model.get_score(importance_type= imp_type)
            except Exception as e:
                print(e)
        return importance

    def get_hist(self, model, used_features):
        hist={}
        for feature in used_features:
            try:
                hist[feature]= model.get_split_value_histogram(feature)
            except Exception as e:
                print(e)
        return hist
    
    def get_metrics_dict(self):
        eval_metrics={}
        eval_metrics['importance']= self.get_importance(self.model)
        eval_metrics['histogram']= self.get_hist(self.model, self.used_features)
        return eval_metrics 