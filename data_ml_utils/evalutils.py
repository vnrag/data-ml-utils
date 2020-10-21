"""A library containing commonly used utils for machine learning evaluation
"""
from data_ml_utils.mainutils import main_utils
from sklearn import metrics
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import json
from data_utils import generalutils as gu
import numpy as np

class general_eval(main_utils):
    logger_name='general_eval'
    conf_matrix= None
    model_metrics={}
    plots={}
    y_actual= None
    y_predicted= None
    y_predicted_prob= None
    feature_lookup= None
    
    def get_atomic_metrics(self, y_actual, y_predicted, y_predicted_prob):
        self.y_actual= y_actual
        self.y_predicted= y_predicted
        self.y_predicted_prob= y_predicted_prob
        self.y_predicted_prob_one= y_predicted_prob
        self.get_perf_metrics()
        self.conf_matrix= self.confusion_matrix()
    
    def get_perf_metrics(self):
        self.atomic_metrics['accuracy']= self.accuracy()
        self.atomic_metrics['f1']= self.f1()
        self.atomic_metrics['precision']= self.precision()
        self.atomic_metrics['recall']= self.recall()
        self.atomic_metrics['auc_roc']= self.roc_auc()
    
    def confusion_matrix(self):
        conf_df= gu.create_data_frame({'actual':self.y_actual,
                          'predicted':self.y_predicted})
        conf_df= conf_df.groupby(['actual','predicted']).size().to_frame('count').reset_index()
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
            self.export_dict_as_one_row_text(self.atomic_metrics, 'atomic_metrics')
            self.export_confusion_matrix_as_text()
        if self.export_s3:
            self.export_atomic_metrics_to_s3()
            self.export_confusion_matrix_to_s3()

    def export_confusion_matrix_as_text(self):
        df= self.conf_matrix
        self.export_df_as_text(df, 'confusion_matrix')
    
    def export_atomic_metrics_to_s3(self):
        df= gu.normalize_json(self.atomic_metrics)
        self.export_metric_to_s3(df, 'atomic_metrics', 'atomic_metrics')
    
    def export_confusion_matrix_to_s3(self):
        df= self.conf_matrix
        self.export_metric_to_s3(df, 'confusion_matrix', 'confusion_matrix')

    def get_roc_plot(self):
        file_path= gu.get_target_path([self.local_folder, 'roc'], file_extension= 'png')
        fpr= self.plots['roc']['fpr']
        tpr= self.plots['roc']['tpr']
        plt.rcParams.update(plt.rcParamsDefault)
        plt.plot(fpr, tpr, marker='.', label='xgboost')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    
    def get_roc_values(self):
        fpr, tpr, _ = metrics.roc_curve(self.y_actual, self.y_predicted_prob_one)
        roc_df= gu.create_data_frame({'fpr':fpr, 'tpr': tpr})
        roc_df.index.name='index'
        roc_df.reset_index(level=0, inplace= True)
        roc_df['model']= self.atomic_metrics['model']
        roc_df['ts']= self.atomic_metrics['ts']
        self.plots['roc']= roc_df
    
    def get_pr_plot(self):
        file_path= gu.get_target_path([self.local_folder, 'pr'], file_extension= 'png')
        plt.rcParams.update(plt.rcParamsDefault)
        precision= self.plots['pr']['precision']
        recall= self.plots['pr']['recall']
        plt.plot(recall, precision, marker='.', label='xgboost')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    
    def get_pr_values(self):
        precision, recall, _ = metrics.precision_recall_curve(self.y_actual, self.y_predicted_prob_one)
        pr_df= gu.create_data_frame({'precision':precision, 'recall': recall})
        pr_df.index.name='index'
        pr_df.reset_index(level=0, inplace= True)
        pr_df['model']= self.atomic_metrics['model']
        pr_df['ts']= self.atomic_metrics['ts']
        self.plots['pr']= pr_df

    def get_prob_plot(self):
        file_path= gu.get_target_path([self.local_folder, 'proba'], file_extension= 'png')
        df= self.plots['prob']
        neg= df[df.classification=='Negatives']['prob_class_1']
        pos= df[df.classification=='Positives']['prob_class_1']
        plt.hist(neg, bins=100, label='Negatives')
        plt.hist(pos, bins=100, label='Positives', alpha=0.7, color='r')
        plt.xlabel('Probability of being Positive Class')
        plt.ylabel('Number of records in each bucket')
        plt.legend()
        plt.tick_params(axis='both', pad=5)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    
    def get_prob_values(self):
        cols = ['predicted','actual','prob_class_1']
        data= np.column_stack([self.y_predicted, self.y_actual, self.y_predicted_prob_one])
        prob_df = gu.create_data_frame(data= data, columns = cols)
        prob_df['classification'] = np.where(prob_df['predicted']==prob_df['actual'],'Positives','Negatives')
        prob_df.index.name='index'
        prob_df.reset_index(level=0, inplace=True)
        prob_df['model']= self.atomic_metrics['model']
        prob_df['ts']= self.atomic_metrics['ts']
        self.plots['prob']= prob_df
    
    def export_roc_as_text(self):
        df= self.plots['roc']
        self.export_df_as_text(df, 'roc_curve')
    
    def export_pr_as_text(self):
        df= self.plots['pr']
        self.export_df_as_text(df, 'pr_curve')
    
    def export_prob_plot_as_text(self):
        df = self.plots['prob']
        self.export_df_as_text(df, 'class_probabilities')    
    
    def export_roc_to_s3(self):
        df= self.plots['roc']
        self.export_metric_to_s3(df, 'roc_curve', 'roc_curve')
    
    def export_pr_to_s3(self):
        df= self.plots['pr']
        self.export_metric_to_s3(df, 'pr_curve', 'pr_curve')

    def export_prob_plot_to_s3(self):
        df = self.plots['prob']
        self.export_metric_to_s3(df, 'class_probabilities', 'class_probabilities')

class xgboost_eval(general_eval):
    logger_name='xgboost_eval'
    model= None
    booster= None
    used_features= None
    validation_metrics=['auc', 'rmse', 'mae', 'logloss', 'error', 'aucpr', 'map']
    
    def get_validation_metrics(self):
        return self.validation_metrics
    
    def get_eval(self, xgb_booster, feature_names):
        self.set_variables(xgb_booster, feature_names)
        self.get_model_metrics()
    
    def set_variables(self, xgb_booster, feature_names):
        # self.model= xgb_model
        self.booster= xgb_booster
        self.feature_lookup= self.get_features_lookup(feature_names)
        self.used_features = xgb_booster.get_score().keys()
    
    def get_features_lookup(self, feature_names):
        df = pd.DataFrame(feature_names, columns=['feature_name'])
        df.index.name= 'feature_code'
        df.reset_index(level=0, inplace=True)
        df['feature_code']= 'f' + df['feature_code'].astype(str)
        return df
    
    def get_model_metrics(self):
        self.model_metrics['importance']= self.get_importance()
        self.model_metrics['histogram']= self.get_hist()
        # self.model_metrics['training_params']= self.get_training_params()
        # self.model_metrics['xgb_specific_params']= self.get_xgb_specific_params()
        # self.model_metrics['validation_results']= self.get_validation_results()
        self.model_metrics['model_config']= self.get_model_config()
    
    def get_importance(self):
        importance_types = ['weight','gain','cover','total_gain','total_cover']
        importance={}
        for imp_type in importance_types:
            try:
                importance[imp_type] = self.booster.get_score(importance_type= imp_type)
            except Exception as e:
                print(e)
        if len(importance) > 0:
            importance_df= gu.create_data_frame(importance)
            importance_df.index.name ='feature'
            importance_df.reset_index(level=0, inplace=True)
            importance_df['model']= self.atomic_metrics['model']
            importance_df['ts']= self.atomic_metrics['ts']
            ### to get feature names
            importance_df= pd.merge(importance_df, self.feature_lookup, left_on='feature', right_on='feature_code').drop('feature_code', axis=1)
            return importance_df
        return

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
        if len(hist) > 0:
            hist_df = pd.concat(hist.values(), ignore_index=True)
            # rearrange columns
            cols = hist_df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            hist_df= hist_df[cols]
            hist_df['mode']= self.atomic_metrics['model']
            hist_df['ts']= self.atomic_metrics['ts']
            ### to get feature names
            hist_df= pd.merge(hist_df, self.feature_lookup, left_on='feature', right_on='feature_code').drop('feature_code', axis=1)
            return hist_df
        return
    
    # def get_training_params(self):
    #     return self.model.get_params()
    
    # def get_xgb_specific_params(self):
    #     return self.model.get_xgb_params()
    
    # def get_validation_results(self):
    #     val_results= self.model.evals_result()
    #     train= gu.create_data_frame(val_results['validation_0'])
    #     train= train.add_prefix('train_')
    #     test= gu.create_data_frame(val_results['validation_1'])
    #     test= test.add_prefix('test_')
    #     train_test= pd.concat([train, test],axis=1)
    #     train_test.index.name='epoch'
    #     train_test.reset_index(level=0, inplace= True)
    #     train_test['model']= self.atomic_metrics['model']
    #     train_test['ts']= self.atomic_metrics['ts']
    #     return train_test
    
    def get_model_config(self):
        return json.loads(self.booster.save_config())

    def export_model_metrics(self):
        # combined_metrics = {**self.atomic_metrics, **self.model_metrics['xgb_specific_params']}
        combined_metrics = self.atomic_metrics
        if self.export_local:
            self.export_dict_as_one_row_text(combined_metrics, 'xgboost_metrics')
            self.export_histogram_as_text()
            self.export_dict_as_text(self.model_metrics['model_config'], 'model_config')
        if self.export_s3:
            self.export_model_metrics_to_s3(combined_metrics)
            self.export_histogram_to_s3()
            self.export_model_config_to_s3()
    
    def export_histogram_as_text(self):
        df= self.model_metrics['histogram']
        self.export_df_as_text(df, 'feature_histogram')
    
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
        self.get_prob_values()
        self.get_pr_values()
        self.get_roc_values()
        if self.export_local:
            self.get_prob_plot()
            self.get_pr_plot()
            self.get_roc_plot()
            # self.get_validation_metrics_plots()
            # self.get_importance_plots()
            # self.get_tree_plot()

    def get_importance_plots(self):
        file_path= gu.get_target_path([self.local_folder, 'imp1'], file_extension= 'png')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.bar(range(len(self.model.feature_importances_)), self.model.feature_importances_)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
        file_path= gu.get_target_path([self.local_folder, 'imp2'], file_extension= 'png')
        plt.rcParams.update(plt.rcParamsDefault)
        xgb.plot_importance(self.model)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    
    def get_tree_plot(self):
        file_path= gu.get_target_path([self.local_folder, 'tree'], file_extension= 'png')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['figure.figsize'] = [50, 10]
        xgb.plot_tree(self.model,num_trees=0)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    
    def get_validation_metrics_plots(self):
        for metric in self.validation_metrics:
            file_path= gu.get_target_path([self.local_folder, f'val_{metric}'], file_extension= 'png')
            df= self.model_metrics['validation_results'][[f'test_{metric}',f'train_{metric}']]
            df.plot()
            plt.ylabel(metric)
            plt.xlabel('epochs')
            plt.title(f'XGBoost {metric}')
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
    
    def export_plots(self):
        if self.export_local:
            # self.export_validation_as_text()
            self.export_importance_as_text()
            # self.export_tree_as_text()
            self.export_roc_as_text()
            self.export_pr_as_text()
            self.export_prob_plot_as_text()
        if self.export_s3:
            # self.export_validation_to_s3()
            self.export_importance_to_s3()
            # self.export_tree_to_s3()
            self.export_roc_to_s3()
            self.export_pr_to_s3()
            self.export_prob_plot_to_s3()
    
    def export_importance_as_text(self):
        df= self.model_metrics['importance']
        self.export_df_as_text(df, 'feature_importance')
    
    def export_tree_as_text(self):
        file_path= gu.get_target_path([self.local_folder, 'tree'], file_extension= 'csv')
        self.booster.dump_model(file_path)
    
    def export_validation_as_text(self):
        df= self.model_metrics['validation_results']
        self.export_df_as_text(df, 'validation_results')

    def export_importance_to_s3(self):
        df= self.model_metrics['importance']
        self.export_metric_to_s3(df, 'feature_importance', 'feature_importance')
    
    # def export_tree_to_s3(self):
    #     self.booster.dump_model('tree.csv')
   
    def export_validation_to_s3(self):
        df= self.model_metrics['validation_results']
        self.export_metric_to_s3(df, 'validation_results', 'validation_results')
