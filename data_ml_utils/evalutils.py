"""A library of utils for machine learning evaluation
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
    """Class for general evaluation utilities
    
    Attributes
    ----------
    conf_matrix : DataFrame
        Confusiuon matrix of testing dataset
    feature_lookup : DataFrame
        Mapping table for features names
    logger_name : str
        Name of the logger
    model_metrics : dict
        Dictionary containing model specific metrics
    plots : dict
        Dictionary containing data for creating plots
    y_actual : ndarray
        Actual target labels
    y_predicted : ndarray
        Predicted target labels
    y_predicted_prob : ndarray
        Probabilities of instances for each class in the model
    y_predicted_prob_one : ndarray
        Probabilities of instances for class== 1
    """

    logger_name='general_eval'
    conf_matrix= None
    model_metrics={}
    plots={}
    y_actual= None
    y_predicted= None
    y_predicted_prob= None
    feature_lookup= None
    
    def get_atomic_metrics(self, y_actual, y_predicted, y_predicted_prob):
        """Calculates general purpose evaluation metrics.
        
        Parameters
        ----------
        y_actual : ndarray
            Actual target labels
        y_predicted : ndarray
            Predicted target labels
        y_predicted_prob : ndarray
            Probabilities of instances for each class in the model
        """
        self.y_actual= y_actual
        self.y_predicted= y_predicted
        self.y_predicted_prob= y_predicted_prob
        self.y_predicted_prob_one= y_predicted_prob
        self.get_perf_metrics()
        self.conf_matrix= self.confusion_matrix()
    
    def get_perf_metrics(self):
        """Calculates general machine learning performance metrics.
        Adds them to atomic_metrics dictionary.
        """
        self.atomic_metrics['accuracy']= self.accuracy()
        self.atomic_metrics['f1']= self.f1()
        self.atomic_metrics['precision']= self.precision()
        self.atomic_metrics['recall']= self.recall()
        self.atomic_metrics['auc_roc']= self.roc_auc()
    
    def confusion_matrix(self):
        """Calculates confusion matrix score of a model
        
        Returns
        -------
        DataFrame
            Confusion Matrix between actual and predicted labels
        """
        conf_df= gu.create_data_frame({'actual':self.y_actual,
                          'predicted':self.y_predicted})
        conf_df= conf_df.groupby(['actual','predicted']).size().to_frame('count').reset_index()
        conf_df['model']= self.atomic_metrics['model']
        conf_df['ts']= self.atomic_metrics['ts']
        return conf_df
    
    def accuracy(self):
        """Calculates accuracy score of a model
        
        Returns
        -------
        float
            Accuracy score from sklearn library
        """
        return metrics.accuracy_score(self.y_actual, self.y_predicted)
    
    def f1(self):
        """Calculates f1 score score of a model
        
        Returns
        -------
        float or array of float
            F1 score from sklearn library
        """
        return metrics.f1_score(self.y_actual, self.y_predicted)
    
    def precision(self):
        """Calculates precision score of a model
        
        Returns
        -------
        float or array of float
            Precision score from sklearn library
        """
        return metrics.precision_score(self.y_actual, self.y_predicted)
    
    def recall(self):
        """Calculates recall score of a model
        
        Returns
        -------
        float or array of float
            Recall score from sklearn
        """
        return metrics.recall_score(self.y_actual, self.y_predicted)
    
    def roc_auc(self):
        """Calculates area under the roc curve score of a model
        
        Returns
        -------
        float
            Area under the roc curve score from sklearn
        """
        return metrics.roc_auc_score(self.y_actual, self.y_predicted)
    
    def export_atomic_metrics(self):
        """Exports atomic metrics and confusion matrix.
        If export_local flag is set to True, metrics are saved to
        text files on the local machine. If export_s3 flag is set
        to True, metrics are saved to parquet files on s3.
        Atomic metrics in text format are exported as one row.
        """
        if self.export_local:
            self.export_dict_as_one_row_text(self.atomic_metrics, 'atomic_metrics')
            self.export_confusion_matrix_as_text()
        if self.export_s3:
            self.export_atomic_metrics_to_s3()
            self.export_confusion_matrix_to_s3()

    def export_confusion_matrix_as_text(self):
        """Exports confusion matrix metric to local disk in csv format
        """
        df= self.conf_matrix
        self.export_df_as_text(df, 'confusion_matrix')
    
    def export_atomic_metrics_to_s3(self):
        """Exports atomic metrics to s3 in parquet format
        """
        df= gu.normalize_json(self.atomic_metrics)
        self.export_metric_to_s3(df, 'atomic_metrics', 'atomic_metrics')
    
    def export_confusion_matrix_to_s3(self):
        """Exports confusion matrix metric to s3 in parquet format
        """
        df= self.conf_matrix
        self.export_metric_to_s3(df, 'confusion_matrix', 'confusion_matrix')

    def get_roc_plot(self):
        """Saves a copy of ROC curve plot on local disk
        """
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
        """Calculates values for creating ROC curve plot.
        Needs true positive rate (TPR) and false positive
        rate (FPR). Stores values in plots dictionary
        """
        fpr, tpr, _ = metrics.roc_curve(self.y_actual, self.y_predicted_prob_one)
        roc_df= gu.create_data_frame({'fpr':fpr, 'tpr': tpr})
        roc_df.index.name='index'
        roc_df.reset_index(level=0, inplace= True)
        roc_df['model']= self.atomic_metrics['model']
        roc_df['ts']= self.atomic_metrics['ts']
        self.plots['roc']= roc_df
    
    def get_pr_plot(self):
        """Saves a copy of PR curve plot on local disk
        """
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
        """Calculates values for creating PR curve plot.
        Needs precision and recall.
        Stores values in plots dictionary
        """
        precision, recall, _ = metrics.precision_recall_curve(self.y_actual, self.y_predicted_prob_one)
        pr_df= gu.create_data_frame({'precision':precision, 'recall': recall})
        pr_df.index.name='index'
        pr_df.reset_index(level=0, inplace= True)
        pr_df['model']= self.atomic_metrics['model']
        pr_df['ts']= self.atomic_metrics['ts']
        self.plots['pr']= pr_df

    def get_prob_plot(self):
        """Saves a copy of class probabilities histogram
        plot on local disk
        """
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
        """Calculates values for creating class probabilities
        histogram plot.
        Needs actual labels, predicted labels and class probabilites
        for class == 1.
        Stores values in plots dictionary
        """
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
        """Exports ROC curve values to local disk in csv format
        """
        df= self.plots['roc']
        self.export_df_as_text(df, 'roc_curve')
    
    def export_pr_as_text(self):
        """Exports PR curve values to local disk in csv format
        """
        df= self.plots['pr']
        self.export_df_as_text(df, 'pr_curve')
    
    def export_prob_plot_as_text(self):
        """Exports class probabilities histogram values
        to local disk in csv format
        """
        df = self.plots['prob']
        self.export_df_as_text(df, 'class_probabilities')    
    
    def export_roc_to_s3(self):
        """Exports ROC curve values to s3 in parquet format
        """
        df= self.plots['roc']
        self.export_metric_to_s3(df, 'roc_curve', 'roc_curve')
    
    def export_pr_to_s3(self):
        """Exports PR curve values to s3 in parquet format
        """
        df= self.plots['pr']
        self.export_metric_to_s3(df, 'pr_curve', 'pr_curve')

    def export_prob_plot_to_s3(self):
        """Exports class probabilities histogram values
        to s3 in parquet format
        """
        df = self.plots['prob']
        self.export_metric_to_s3(df, 'class_probabilities', 'class_probabilities')

class xgboost_eval(general_eval):

    """Class for XGBoost model evaluation utilities
    
    Attributes
    ----------
    booster : Booster
        A Booster of XGBoost
    feature_lookup : DataFrame
        Mapping of features to their names
    hyper_params : list
        Hyperparameters of the model
    logger_name : str
        Name of Logger
    model : XGBClassifier
        XGBoost implementation of sklearn API for XGBoost classification
    used_features : list
        Features used for building the model
    validation_metrics : list
        Metrics used for the evluation during learning step
    """
    
    logger_name='xgboost_eval'
    model= None
    booster= None
    used_features= None
    validation_metrics=['auc', 'rmse', 'mae', 'logloss', 'error', 'aucpr', 'map']
    hyper_params=['alpha','eta','gamma','lambda','max_delta_step','max_depth','min_child_weight']
    
    def get_validation_metrics(self):
        """Retrieves a list of xgboost model 
        validation metrics
        
        Returns
        -------
        list
            Metrics for evaluation during learning step
        """
        return self.validation_metrics
    
    def get_eval(self, xgb_booster, feature_names):
        """Evaluates xgboost model and store metrics in
        model_metrics dictionary
        
        Parameters
        ----------
        xgb_booster : Booster
            A Booster of XGBoost
        feature_names : list
            Featues names
        """
        self.set_variables(xgb_booster, feature_names)
        self.get_model_metrics()
    
    def set_variables(self, xgb_booster, feature_names):
        """Stores xgboost model object, extracts used
        features and creates features names mapping
        table.
        
        Parameters
        ----------
        xgb_booster : Booster
            A Booster of XGBoost
        feature_names : list
           Features names
        """
        # self.model= xgb_model
        self.booster= xgb_booster
        self.feature_lookup= self.get_features_lookup(feature_names)
        self.used_features = xgb_booster.get_score().keys()
    
    def get_features_lookup(self, feature_names):
        """Creates a mapping table between xgboost model
        features and their names
        
        Parameters
        ----------
        feature_names : list
            Features Names
        
        Returns
        -------
        DataFrame
            Mapping table for features and their names
        """
        df = pd.DataFrame(feature_names, columns=['feature_name'])
        df.index.name= 'feature_code'
        df.reset_index(level=0, inplace=True)
        df['feature_code']= 'f' + df['feature_code'].astype(str)
        return df
    
    def get_model_metrics(self):
        """Calculates xgboost specific performance metrics.
        Adds them to model_metrics dictionary.
        """
        self.model_metrics['importance']= self.get_importance()
        self.model_metrics['histogram']= self.get_hist()
        # self.model_metrics['training_params']= self.get_training_params()
        # self.model_metrics['xgb_specific_params']= self.get_xgb_specific_params()
        # self.model_metrics['validation_results']= self.get_validation_results()
        # self.model_metrics['model_config']= self.get_model_config()
        ## hyper params are just subset of model config
        self.model_metrics['hyper_params']= self.get_hyperparams()
    
    def get_hyperparams(self):
        """Extracts hyperparameters from xgboost model
        
        Returns
        -------
        DataFrame
            Hyperparameters of learned model
        """
        hp={}
        config= json.loads(self.booster.save_config())
        train_params= config['learner']['gradient_booster']['updater']['grow_colmaker']['train_param']
        for param in self.hyper_params:
            try:
                hp[param]= train_params[param]
            except Exception as e:
                print('Error While getting hyperparameters')
                print(f'Parameter {param}: {e}')
        if len(hp) > 0:
            hp_sr= pd.to_numeric(pd.Series(hp,name='value'))
            hp_sr.index.name='parameter'
            hp_df= hp_sr.to_frame().reset_index()
            hp_df['model']= self.atomic_metrics['model']
            hp_df['ts']= self.atomic_metrics['ts']
            return hp_df
        return
    
    def get_importance(self):
        """Extracts features importance from xgboost model
        Uses different importance criteria from the xgboost
        library
        Returns
        -------
        DataFrame
            Features importance table
        """
        importance_types = ['weight','gain','cover','total_gain','total_cover']
        importance={}
        for imp_type in importance_types:
            try:
                importance[imp_type] = self.booster.get_score(importance_type= imp_type)
            except Exception as e:
                print('Error While getting feature importance')
                print(f'Importance Type {imp_type}: {e}')
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
        """Extracts split values histogram from xgboost 
        model.
        
        Returns
        -------
        DataFrame
            Split values histogram table
        """
        hist={}
        for feature in self.used_features:
            try:
                df= self.booster.get_split_value_histogram(feature)
                df.index.name= 'split_index'
                df.reset_index(level=0, inplace=True)
                df['feature']= feature
                hist[feature]= df
            except Exception as e:
                print('Error While getting feature histrogram')
                print(f'Feature {feature}: {e}')
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
    #     """Extracts training parameters from xgboost model
        
    #     Returns
    #     -------
    #     dict
    #         Training parameters
    #     """
    #     return self.model.get_params()
    
    # def get_xgb_specific_params(self):
    #     """Extracts specific training parameters from
    #     xgboost model
        
    #     Returns
    #     -------
    #     dict
    #         Xgboost model specific parameters
    #     """
    #     return self.model.get_xgb_params()
    
    # def get_validation_results(self):
    #     """Extracts results of data validation step
    #     from xgboost model
        
    #     Returns
    #     -------
    #     DataFrame
    #         Validation results table
    #     """
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
        """Extracts internal parameters configuration
        of xgboost model
        
        Returns
        -------
        dict
            Internal parameters configuration
        """
        return json.loads(self.booster.save_config())

    def export_model_metrics(self):
        """Exports each xgboost model metric to a separate file.
            If export_local flag is set to True, metrics are saved to
            text files on the local machine. If export_s3 flag is set
            to True, metrics are saved to parquet files on s3.
        """
        # combined_metrics = {**self.atomic_metrics, **self.model_metrics['xgb_specific_params']}
        combined_metrics = self.atomic_metrics
        if self.export_local:
            # self.export_dict_as_one_row_text(combined_metrics, 'xgboost_metrics')
            # self.export_histogram_as_text()
            # self.export_dict_as_text(self.model_metrics['model_config'], 'model_config')
            self.export_hyperparameters_as_text()
        if self.export_s3:
            # self.export_model_metrics_to_s3(combined_metrics)
            # self.export_histogram_to_s3()
            # self.export_model_config_to_s3()
            self.export_hyperparameters_to_s3()
    
    def export_hyperparameters_as_text(self):
        """Exports xgboost hyperparameters to local disk
        in csv format
        """
        try:
            df= self.model_metrics['hyper_params']
            self.export_df_as_text(df, 'hyperparameters')
        except Exception as e:
            print('Error While exporting Hyperparameters to local disk')
            print(f'Hyperparameters: {e}')
    
    def export_hyperparameters_to_s3(self):
        """Exports xgboost hyperparameters to s3 in parquet format
        """
        try:
            df= self.model_metrics['hyper_params']
            self.export_metric_to_s3(df, 'hyperparameters', 'hyperparameters')
        except Exception as e:
            print('Error While exporting Hyperparameters to s3')
            print(f'Hyperparameters: {e}')
    
    def export_histogram_as_text(self):
        """Exports xgboost features histogram to local disk
        in csv format
        """
        try:
            df= self.model_metrics['histogram']
            self.export_df_as_text(df, 'feature_histogram')
        except Exception as e:
            print('Error While exporting Features Histogram to local disk')
            print(f'Features Histogram: {e}')
    
    def export_model_metrics_to_s3(self, combined_metrics):
        """Exports atomic and xgboost metrics together
        
        Parameters
        ----------
        combined_metrics : dict
            Atomic and xgboost metrics
        """
        df= gu.normalize_json(combined_metrics)
        self.export_metric_to_s3(df, 'xgboost_metrics', 'xgboost_metrics')
    
    def export_histogram_to_s3(self):
        """Exports xgboost features histogram to s3 in
        parquet format
        """
        try:
            df= self.model_metrics['histogram']
            self.export_metric_to_s3(df, 'feature_histogram', 'feature_histogram')
        except Exception as e:
            print('Error While exporting Features Histogram to s3')
            print(f'Features Histogram: {e}')
    
    def export_model_config_to_s3(self):
        """Exports internal parameters configuration of
        xgboost model to s3 in parquet format
        """
        df= gu.normalize_json(self.model_metrics['model_config'])
        self.export_metric_to_s3(df, 'model_config', 'model_config')

    def get_plots(self):
        """Creates data for xbgoost evaluation plots.
        Exports images to local disk it export_local flag
        is set to True
        """
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
        """Prepares xgboost features importance bar plots.
        Saves images to local disk
        """
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
        """Prepares xgboost model decision tree diagram.
        Saves images to local disk.
        """
        file_path= gu.get_target_path([self.local_folder, 'tree'], file_extension= 'png')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['figure.figsize'] = [50, 10]
        xgb.plot_tree(self.model,num_trees=0)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
    
    def get_validation_metrics_plots(self):
        """Prepares xgboost validation metrics plots.
        Saves images to local disk
        """
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
        """Exports xgboost evaluation plots data.
        If export_local flag is set to True, data is saved to
        text files on the local machine. If export_s3 flag is set
        to True, data is saved to parquet files on s3.
        """
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
        """Exports xgboost features importance metric to
        local disk in csv format
        """
        try:
            df= self.model_metrics['importance']
            self.export_df_as_text(df, 'feature_importance')
        except Exception as e:
            print('Error While exporting Features Importace to local disk')
            print(f'feature importance: {e}')
    
    def export_tree_as_text(self):
        """Exports xgboost decision tree to local 
        disk in csv format
        """
        file_path= gu.get_target_path([self.local_folder, 'tree'], file_extension= 'csv')
        self.booster.dump_model(file_path)
    
    def export_validation_as_text(self):
        """Exports xgboost validation results to
        local disk in csv format
        """
        df= self.model_metrics['validation_results']
        self.export_df_as_text(df, 'validation_results')

    def export_importance_to_s3(self):
        """Exports xgboost features importance metric to
        s3 in parquet format
        """
        try:
            df= self.model_metrics['importance']
            self.export_metric_to_s3(df, 'feature_importance', 'feature_importance')
        except Exception as e:
            print('Error While exporting Features Importace to s3')
            print(f'feature importance: {e}')
    
    # def export_tree_to_s3(self):
    #     """Exports xgboost decision tree to
    #     s3 in parquet format
    #     """
    #     self.booster.dump_model('tree.csv')
   
    def export_validation_to_s3(self):
        """Exports xgboost validation results to
        s3 in parquet format
        """
        df= self.model_metrics['validation_results']
        self.export_metric_to_s3(df, 'validation_results', 'validation_results')
