# -*- coding: utf-8 -*-
"""FINAL_ML_Research_stage1.ipynb

Original file is located at
    https://colab.research.google.com/drive/1HoHLKGHd2kFPuZzLDbeeGhJnP9SILmbJ

Import Libraries
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import json
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from main_MAF_classification import maf_train, eval
from main_GMM_classification import train as gmm_train, eval as gmm_eval
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold, train_test_split
import pickle
from sklearn.model_selection import cross_val_score
import xgboost as xgb

"""Configurations"""

# Define the configurations
maf_config = {
    "stage": "train",
    "train": {
        "data_type": "real_data",
        "real_data": {
            "train_data_path": "./dataset/SAHeart/SAHeart.csv",
            "test_data_path": None,
            "val_data_path": None,
            "split_data_again": True,
            "features": ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol", "age"],
            "label_feature": "chd",
            "categorical_features": ["famhist"]
        },
        "toy_data": {
            "dataset_name": "circles",
            "dataset_size": 2000,
            "n_classes": 2
        },
        "log_path": "./log_train/",
        "batch_size": 800,
        "hidden_shape": [30, 30, 30, 30, 30, 30, 30, 30],
        "layers": 20,
        "base_lr": 1e-5,
        "end_lr": 1e-6,
        "max_epochs": 5e3,
        "delta_stop_in_early_stopping": 1000,
        "frequency_validation": 100,
        "frequency_plot": 1000,
        "plot_data": False
    },
    "eval": {
        "log_path": "./log_eval/",
        "use_posterior": True
    }
}

gmm_config = {
    "stage": "train",
    "train": {
        "data_type": "real_data",
        "real_data": {
            "train_data_path": "./dataset/SAHeart/SAHeart.csv",
            "test_data_path": None,
            "val_data_path": None,
            "split_data_again": True,
            "features": ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol", "age"],
            "label_feature": "chd",
            "categorical_features": ["famhist"]
        },
        "toy_data": {
            "dataset_name": "circles",
            "dataset_size": 2000,
            "n_classes": 2
        },
        "log_path": "./log_train/",
        "batch_size": 800,
        "n_components": 3,
        "plot_data": False
    },
    "eval": {
        "log_path": "./log_eval/",
        "use_posterior": True
    }
}

"""Function load dataset generator"""

def load_dataset_generator(config: Dict, split_data_again: bool):
    # read data:
    path = config['train']['log_path']+f"data/"
    # read data:
    df = pd.read_csv(config['train']['real_data']['train_data_path'])
    X = df[config['train']['real_data']['features']]
    y = df[config['train']['real_data']['label_feature']]
    # split data to train, test, and val:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42) # 0.25 x 0.8 = 0.2
#         print(X_train.shape)
        # numerical and categorical features:
        categorical_features = config['train']['real_data']['categorical_features']
        numeric_features = [col for col in config['train']['real_data']['features'] if col not in categorical_features]

        # preprocess data:
        numeric_transformer = Pipeline(
            steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]
        )
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    #     categorical_transformer = ce.BackwardDifferenceEncoder(drop_invariant=True)
    #     categorical_transformer = ce.helmert.HelmertEncoder(drop_invariant=True)
        data_pipeline = ColumnTransformer([
            ('numerical', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ])
        X_train_processed = data_pipeline.fit_transform(X_train)
        X_test_processed = data_pipeline.transform(X_test)
        X_val_processed = data_pipeline.transform(X_val)

        # convert data to float32 --> important: otherwise, the network gives error!
        X_train_processed = X_train_processed.astype(np.float32)
        X_test_processed = X_test_processed.astype(np.float32)
        X_val_processed = X_val_processed.astype(np.float32)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        y_val = y_val.values.ravel()

        # split to classes:
        batched_train_data_list, train_data_list, val_data_list, test_data_list = [], [], [], []
        n_classes = len(np.unique(y_train))
        for class_index in range(n_classes):
            X_train_processed_class = X_train_processed[y_train==class_index, :]
            X_test_processed_class = X_test_processed[y_test==class_index, :]
            X_val_processed_class = X_val_processed[y_val==class_index, :]

            # make batched train data:
            batched_train_data = tf.data.Dataset.from_tensor_slices(X_train_processed_class)
            BATCH_SIZE = config['train']['batch_size']
            SHUFFLE_BUFFER_SIZE = 100
            batched_train_data = batched_train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

            # make the list:
            batched_train_data_list.append(batched_train_data)
            train_data_list.append(X_train_processed_class)
            test_data_list.append(X_test_processed_class)
            val_data_list.append(X_val_processed_class)
        yield batched_train_data_list, train_data_list, val_data_list, test_data_list

"""Cross Validation"""

# Cross-Validation for MAF

maf_results = []

Data_gen = load_dataset_generator(config = maf_config, split_data_again = maf_config['train']['real_data']['split_data_again'])

for data_iter in Data_gen:
    maf_train(maf_config, data_iter)
    maf_result = eval(maf_config)[3:5]
    maf_results.append(maf_result)


# Cross-Validation for GMM

gmm_results = []

Data_gen = load_dataset_generator(config=gmm_config, split_data_again=gmm_config['train']['real_data']['split_data_again'])

for data_iter in Data_gen:
    gmm_train(gmm_config, data_iter)
    result = gmm_eval(gmm_config)[3:5]
    gmm_results.append(result)



"""ML Models"""

config = {
    "stage": "train",
    "train": {
        "data_type": "real_data",
        "real_data": {
            "train_data_path": "./dataset/SAHeart/SAHeart.csv",
            "test_data_path": None,
            "val_data_path": None,
            "split_data_again": True,
            "features": ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol", "age"],
            "label_feature": "chd",
            "categorical_features": ["famhist"]
         },
        "log_path": "./log_train/",
        "batch_size": 32,
        "random_seed": 42,
    }
}

def load_SAHeart_dataset_and_train_MLs():
    # read data:
    path = config['train']['log_path']+f"data/"
    # read data:
    df = pd.read_csv(config['train']['real_data']['train_data_path'])
    X = df[config['train']['real_data']['features']]
    y = df[config['train']['real_data']['label_feature']]
    # split data to train, test, and val:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    svm_performance = []
    lr_performance = []
    rf_performance = []
    mlp_performance = []
    lda_performance = []
    xgb_performance = []
    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42) # 0.25 x 0.8 = 0.2
#         print(X_train.shape)
        # numerical and categorical features:
        categorical_features = config['train']['real_data']['categorical_features']
        numeric_features = [col for col in config['train']['real_data']['features'] if col not in categorical_features]

        # preprocess data:
        numeric_transformer = Pipeline(
            steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]
        )
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
#         categorical_transformer = ce.BackwardDifferenceEncoder(drop_invariant=True)
    #     categorical_transformer = ce.helmert.HelmertEncoder(drop_invariant=True)
        data_pipeline = ColumnTransformer([
            ('numerical', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ])
        X_train_processed = data_pipeline.fit_transform(X_train)
        X_test_processed = data_pipeline.transform(X_test)
        X_val_processed = data_pipeline.transform(X_val)

        # convert data to float32 --> important: otherwise, the network gives error!
        X_train_processed = X_train_processed.astype(np.float32)
        X_test_processed = X_test_processed.astype(np.float32)
        X_val_processed = X_val_processed.astype(np.float32)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        y_val = y_val.values.ravel()
        
        #SVM
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train_processed, y_train)
        y_pred = svm_model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        svm_performance.append((accuracy, f1))
        #LR
        logistic_model = LogisticRegression(random_state=42)
        logistic_model.fit(X_train_processed, y_train)
        y_pred = logistic_model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        lr_performance.append((accuracy, f1))            
        #RF
        rf_model = RandomForestClassifier(n_estimators=100)
        rf_model.fit(X_train_processed, y_train)
        y_pred = rf_model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        rf_performance.append((accuracy, f1))   
        #MLP
        mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        mlp_model.fit(X_train_processed, y_train)
        y_pred = mlp_model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlp_performance.append((accuracy, f1))
        #LDA
        lda_model = LDA(store_covariance=True, solver='lsqr')
        lda_model.fit(X_train_processed, y_train)
        y_pred = rf_model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        lda_performance.append((accuracy, f1))
        #XGB
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train_processed, y_train)
        y_pred = xgb_model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        xgb_performance.append((accuracy, f1))
    
    with open('model_performances.txt', 'a') as f:
        f.write(f"SVM: {svm_performance}\n")
        f.write(f"SVM: {[sum(x) / len(svm_performance) for x in zip(*svm_performance)]}\n")
        f.write(f"Logistic Regression: {lr_performance}\n")
        f.write(f"Logistic Regression: {[sum(x) / len(lr_performance) for x in zip(*lr_performance)]}\n")
        f.write(f"RF: {rf_performance}\n")
        f.write(f"RF: {[sum(x) / len(rf_performance) for x in zip(*rf_performance)]}\n")
        f.write(f"MLP: {mlp_performance}\n")
        f.write(f"MLP: {[sum(x) / len(mlp_performance) for x in zip(*mlp_performance)]}\n")
        f.write(f"LDA: {lda_performance}\n")
        f.write(f"LDA: {[sum(x) / len(lda_performance) for x in zip(*lda_performance)]}\n")
        f.write(f"XGB: {xgb_performance}\n")
        f.write(f"XGB: {[sum(x) / len(xgb_performance) for x in zip(*xgb_performance)]}\n")
        


with open('model_performances.txt', 'w') as f:
    f.write(f"GMM Results (Accuracy, F1-score): {gmm_results}\n")
    f.write(f"GMM Results (Accuracy, F1-score): {[sum(x) / len(gmm_results) for x in zip(*gmm_results)]}\n")
    f.write(f"MAF Results (Accuracy, F1-score): {maf_results}\n")
    f.write(f"MAF Results (Accuracy, F1-score): {[sum(x) / len(maf_results) for x in zip(*maf_results)]}\n")
    
load_SAHeart_dataset_and_train_MLs()
