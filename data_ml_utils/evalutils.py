"""A library containing commonly used utils for machine learning evaluation
"""
import logging
from sklearn import metrics
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import csv

class general_eval(object):
    eval_type="general_eval"
    logger=None
    atomic_metrics={}
    conf_matrix= None
    model_metrics={}
    y_actual= None
    y_predicted= None
    y_predicted_prob= None
    
    def __init__(self, algorithm_name, timestamp, train_data_name, num_rows, num_features, cv_folds):
        self.logger= logging.getLogger(self.eval_type)
        self.logger.setLevel(logging.CRITICAL)
        self.atomic_metrics['algorithm_name']= algorithm_name
        self.atomic_metrics['timestamp']= timestamp
        self.atomic_metrics['train_data_name']= train_data_name
        self.atomic_metrics['num_rows']= num_rows
        self.atomic_metrics['num_features']= num_features
        self.atomic_metrics['cv_folds']= cv_folds
        
    
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
       return pd.crosstab(self.y_actual, self.y_predicted, \
                        rownames=['Actual'], colnames=['Predicted'], \
                        margins=True, margins_name='Total')
    
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
    
    def export_atomic_metrics_as_text(self):
        self.save_dict_as_text(self.atomic_metrics, 'atomic_metrics')
        self.save_dict_as_text(self.conf_matrix, 'confusion_matrix')
    
    def save_dict_as_text(self, data_dict , fname):
        with open(f'{fname}.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data_dict.items():
                writer.writerow([key, value])

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
        return importance

    def get_hist(self):
        hist={}
        for feature in self.used_features:
            try:
                hist[feature]= self.booster.get_split_value_histogram(feature)
            except Exception as e:
                print(e)
        return hist
    
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
        return train_test
    
    def get_model_config(self):
        return json.loads(self.booster.save_config())

    def get_plots(self):
        self.get_validation_metrics_plots()
        self.get_importance_plots()
        self.get_tree_plot()
        self.get_roc_plot()
        self.get_pr_plot()
        
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
        roc={}
        plt.rcParams.update(plt.rcParamsDefault)
        roc['fpr'], roc['tpr'], _ = metrics.roc_curve(self.y_actual, self.y_predicted_prob_one)
        plt.plot(roc['fpr'], roc['tpr'], marker='.', label='xgboost')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('roc.png', bbox_inches='tight')
        plt.close()
        self.plots['roc']=roc
    
    def get_pr_plot(self):
        pr={}
        plt.rcParams.update(plt.rcParamsDefault)
        pr['precision'], pr['recall'], _ = metrics.precision_recall_curve(self.y_actual, self.y_predicted_prob_one)
        plt.plot(pr['recall'], pr['precision'], marker='.', label='xgboost')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig('pr.png', bbox_inches='tight')
        plt.close()
        self.plots['pr']=pr
    
    def get_validation_metrics_plots(self):
        for metric in self.validation_metrics:
            df= self.model_metrics['validation_results'][[f'test_{metric}',f'train_{metric}']]
            df.plot()
            plt.ylabel(metric)
            plt.xlabel('epochs')
            plt.title(f'XGBoost {metric}')
            plt.savefig(f'val_{metric}.png', bbox_inches='tight')
            plt.close()

    def export_plots_as_text(self):
        self.export_importance_as_text()
        self.export_tree_as_text()
        self.export_roc_as_text()
        self.export_pr_as_text()
    
    def export_importance_as_text(self):
        df= pd.DataFrame(self.model_metrics['importance'])
        df.index.name ='feature'
        df.reset_index(level=0, inplace=True)
        df.to_csv(r'imp.csv', index=False, header=True)
    
    def export_tree_as_text(self):
        self.booster.dump_model('tree.csv')
    
    def export_roc_as_text(self):
        df= pd.DataFrame({'fpr':self.plots['roc']['fpr'], 'tpr': self.plots['roc']['tpr']})
        df.to_csv(r'roc.csv', index = False, header=True)
    
    def export_pr_as_text(self):
        df= pd.DataFrame({'precision':self.plots['pr']['precision'], 'recall': self.plots['pr']['recall']})
        df.to_csv(r'pr.csv', index = False, header=True)
    
    def export_validation_as_text(self):
        df= self.model_metrics['validation_results']
        df.to_csv(r'validation_results.csv', index = False, header=True)
    
    def export_model_metrics_as_text(self):
        pass
        # self.ave_dict_as_text(self.model_metrics, 'model_metrics')