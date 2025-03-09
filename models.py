import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from pysurvival.models.multi_task import LinearMultiTaskModel

import torch
import torchtuples as tt
from pycox.models import DeepHitSingle, mtlr
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from config import SEED as seed
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np
from sksurv.svm import FastSurvivalSVM
from lifelines import WeibullAFTFitter
from itertools import product
from config import best_params_ds, best_params_deephit, best_params_nmtlr, best_params_xgb


def xgboost_cv(X, y_df, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42)):
    best_score = -np.inf
    best_params = None
    scores = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    # Iterate through all combinations of parameters in the parameter grid
    for param_comb in product(*values):
        params = dict(zip(keys, param_comb))
        fold_scores = []
        
        # Perform cross-validation
        for train_idx, val_idx in cv.split(X):
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y_df.iloc[train_idx]
            y_val = y_df.iloc[val_idx]
            
            # Prepare data for XGBoost (ensure correct format for y: survival times and events)
            dtrain = xgb.DMatrix(X_tr, label=y_tr['time'].values)
            dval = xgb.DMatrix(X_val, label=y_val['time'].values)

            # Set up the XGBoost model with the provided hyperparameters
            model = xgb.XGBModel(objective='survival:cox', eval_metric='cox-nloglik', **params)

            # Train the model
            model.fit(X_tr, y_tr['time'], eval_set=[(X_val, y_val['time'])], early_stopping_rounds=50, verbose=False)

            # Predict the risk scores on validation set
            preds = model.predict(X_val)

            # Calculate the C-index for survival prediction
            c_index = concordance_index_censored(y_val['status'].astype(bool), y_val['time'], preds)[0]
            fold_scores.append(c_index)
        
        # Calculate the average C-index score across all folds
        avg_score = np.mean(fold_scores)
        scores.append((params, avg_score))
        print(f"XGBoost params: {params} | Avg C-index: {avg_score:.4f}")

        # Update best parameters if the current combination gives a better score
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    return best_params, best_score, scores


# DeepHit超参数搜索
def deephit_cv(X, y_df, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=seed)):
    # num_durations=10
    from itertools import product
    best_score = -np.inf
    best_params = None
    scores = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for param_comb in product(*values):
        params = dict(zip(keys, param_comb))
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y_df.iloc[train_idx]
            y_val = y_df.iloc[val_idx]

            labtrans = LabTransDiscreteTime(params['num_durations'])
            durations_tr = y_tr['time'].values
            events_tr = y_tr['status'].values.astype(bool)
            y_tr_deephit = labtrans.fit_transform(durations_tr, events_tr)

            net = tt.practical.MLPVanilla(X_tr.shape[1], params['num_nodes'], labtrans.out_features,
                                        batch_norm=True, dropout=params['dropout'])
            optimizer = tt.optim.Adam(lr=params['lr'])
            model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
            model.set_device('cpu')
            model.fit(X_tr.values.astype('float32'), y_tr_deephit,
                    batch_size=params['batch_size'], epochs=params['num_epochs'], verbose=False)

            surv_df = model.predict_surv_df(X_val.values.astype('float32'))
            t0 = np.median(y_val['time'])
            available_times = np.array(surv_df.index, dtype=float)
            closest_time = available_times[np.argmin(np.abs(available_times - t0))]
            pred = 1 - surv_df.loc[closest_time].values

            c_index = concordance_index_censored(y_val['status'].astype(bool), y_val['time'], pred)[0]
            fold_scores.append(c_index)
        
        avg_score = np.mean(fold_scores)
        scores.append((params, avg_score))
        print(f"DeepHit params: {params} | Avg C-index: {avg_score:.4f}")
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    return best_params, best_score, scores

# (3) DeepSurv 超参数调优
def deep_surv_cv(X, y_df, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=seed)):
    from itertools import product
    best_score = -np.inf
    best_params = None
    scores = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for param_comb in product(*values):
        params = dict(zip(keys, param_comb))
        fold_scores = []
        for train_idx, val_idx in cv.split(X):
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y_df.iloc[train_idx]
            y_val = y_df.iloc[val_idx]
            # 转换为结构化数组
            y_tr_struct = Surv.from_arrays(event=y_tr['status'], time=y_tr['time'])
            y_val_struct = Surv.from_arrays(event=y_val['status'], time=y_val['time'])
            model = LinearMultiTaskModel()  # 每个折都重新实例化模型
            model.fit(X_tr, y_tr_struct['time'], y_tr_struct['event'], 
                    init_method='orthogonal', optimizer=params['optimizer'],
                    lr=params['lr'], num_epochs=params['num_epochs'], verbose=False)
            risk = model.predict_risk(X_val)
            c_index = concordance_index_censored(y_val_struct['event'], y_val_struct['time'], risk)[0]
            fold_scores.append(c_index)
        avg_score = np.mean(fold_scores)
        scores.append((params, avg_score))
        print("参数组合：{} 平均 C-index: {:.4f}".format(params, avg_score))
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
        return best_params, best_score, scores


    # NMTLR超参数搜索
def nmtlr_cv(X, y_df, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=seed)):
    from itertools import product
    best_score = -np.inf
    best_params = None
    scores = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for param_comb in product(*values):
        params = dict(zip(keys, param_comb))
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y_df.iloc[train_idx]
            y_val = y_df.iloc[val_idx]

            labtrans = LabTransDiscreteTime(params['num_durations'])
            durations_tr = y_tr['time'].values
            events_tr = y_tr['status'].values.astype(bool)
            y_tr_nmtlr = labtrans.fit_transform(durations_tr, events_tr)

            net = tt.practical.MLPVanilla(X_tr.shape[1], params['num_nodes'], labtrans.out_features,
                                        batch_norm=True, dropout=params['dropout'])
            optimizer = tt.optim.Adam(lr=params['lr'])
            model = mtlr.MTLR(net, optimizer, duration_index=labtrans.cuts)
            model.set_device('cpu')
            model.fit(X_tr.values.astype('float32'), y_tr_nmtlr,
                    batch_size=params['batch_size'], epochs=params['num_epochs'], verbose=False)

            surv_df = model.predict_surv_df(X_val.values.astype('float32'))
            t0 = np.median(y_val['time'])
            available_times = np.array(surv_df.index, dtype=float)
            closest_time = available_times[np.argmin(np.abs(available_times - t0))]
            pred = 1 - surv_df.loc[closest_time].values

            c_index = concordance_index_censored(y_val['status'].astype(bool), y_val['time'], pred)[0]
            fold_scores.append(c_index)
        
        avg_score = np.mean(fold_scores)
        scores.append((params, avg_score))
        print(f"NMTLR params: {params} | Avg C-index: {avg_score:.4f}")
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    return best_params, best_score, scores
    
def train_base_models(X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, y_train_df, y_test_df):
    models = {}
    
    # CoxPH模型（保持原样）
    train_data = pd.concat([X_train.reset_index(drop=True),
                        pd.DataFrame({'time': y_train['time'], 'status': y_train['event']})], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_data, duration_col='time', event_col='status')
    models['CoxPH'] = cph
    
    # 随机生存森林（保持原样）
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, 
                            min_samples_leaf=10, n_jobs=-1, random_state=seed)
    rsf.fit(X_train_scaled, y_train)
    models['RSF'] = rsf
    
    # 梯度提升生存分析（保持原样）
    gbs = GradientBoostingSurvivalAnalysis(loss="coxph", learning_rate=0.1,
                                        n_estimators=100, random_state=seed)
    gbs.fit(X_train_scaled, y_train)
    models['GBSA'] = gbs


    # # 定义 DeepSurv 参数搜索网格
    # param_grid_ds = {
    #     'lr': [1e-3, 1e-4],
    #     'num_epochs': [200, 400],
    #     'optimizer': ['adam']
    # }

    # # print("开始 DeepSurv 超参数调优……")
    # # best_params_ds, best_score_ds, ds_scores = deep_surv_cv(X_train_scaled, y_train_df, param_grid_ds)
    # # print("DeepSurv 最佳超参数：", best_params_ds, "对应 C-index：", best_score_ds)

    # # 用最佳超参数在全训练集上重新训练 DeepSurv 模型
    # ds_model = LinearMultiTaskModel()
    # ds_model.fit(X_train_scaled, y_train['time'], y_train['event'], 
    #             init_method='orthogonal', optimizer=best_params_ds['optimizer'],
    #             lr=best_params_ds['lr'], num_epochs=best_params_ds['num_epochs'], verbose=False)
    # models['DeepSurv'] = ds_model



    # # 定义参数网格
    # param_grid_deephit = {
    #     'lr': [1e-3, 1e-4],
    #     'num_epochs': [200],
    #     'batch_size': [256, 512],
    #     'num_nodes': [[64, 64], [32, 32]],
    #     'dropout': [0.1, 0.2],
    #     'num_durations': [12, 24]
    # }

    # param_grid_nmtlr = {
    #     'lr': [1e-3, 1e-4],
    #     'num_epochs': [200],
    #     'batch_size': [256, 512],
    #     'num_nodes': [[32, 32], [64, 64]],
    #     'dropout': [0.1, 0.2],
    #     'num_durations': [12, 24]
    # }

    # # 执行DeepHit参数搜索
    # print("Starting DeepHit hyperparameter search...")
    # best_params_deephit, best_score_deephit, _ = deephit_cv(X_train_scaled, y_train_df, param_grid_deephit)
    # print(f"Best DeepHit params: {best_params_deephit} | Best C-index: {best_score_deephit:.4f}")
    # best_params_deephit = {'lr': 0.0001, 'num_epochs': 200, 'batch_size': 256, 'num_nodes': [64, 64], 'dropout': 0.2, 'num_durations': 120} 
    # best_params_nmtlr = {'lr': 0.001, 'num_epochs': 200, 'batch_size': 512, 'num_nodes': [32, 32], 'dropout': 0.2, 'num_durations': 120}
    # 使用最佳参数训练DeepHit
    num_durations = best_params_deephit['num_durations']
    labtrans = LabTransDiscreteTime(num_durations)
    durations = y_train_df['time'].values
    events = y_train_df['status'].values.astype(bool)
    y_train_deephit = labtrans.fit_transform(durations, events)

    net = tt.practical.MLPVanilla(X_train_scaled.shape[1], best_params_deephit['num_nodes'], 
                                labtrans.out_features, batch_norm=True, 
                                dropout=best_params_deephit['dropout'])
    optimizer = tt.optim.Adam(lr=best_params_deephit['lr'])
    model_deephit = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
    model_deephit.set_device('cpu')
    model_deephit.fit(X_train_scaled.values.astype('float32'), y_train_deephit,
                    batch_size=best_params_deephit['batch_size'],
                    epochs=best_params_deephit['num_epochs'], verbose=False)
    models['DeepHit'] = model_deephit

    # # 执行NMTLR参数搜索
    # print("Starting NMTLR hyperparameter search...")
    # best_params_nmtlr, best_score_nmtlr, _ = nmtlr_cv(X_train_scaled, y_train_df, param_grid_nmtlr)
    # print(f"Best NMTLR params: {best_params_nmtlr} | Best C-index: {best_score_nmtlr:.4f}")

    # 使用最佳参数训练NMTLR
    num_durations = best_params_nmtlr['num_durations']
    labtrans = LabTransDiscreteTime(num_durations)
    durations = y_train_df['time'].values
    events = y_train_df['status'].values.astype(bool)
    y_train_nmtlr = labtrans.fit_transform(durations, events)

    net = tt.practical.MLPVanilla(X_train_scaled.shape[1], best_params_nmtlr['num_nodes'], 
                                labtrans.out_features, batch_norm=True, 
                                dropout=best_params_nmtlr['dropout'])
    optimizer = tt.optim.Adam(lr=best_params_nmtlr['lr'])
    model_nmtlr = mtlr.MTLR(net, optimizer, duration_index=labtrans.cuts)
    model_nmtlr.set_device('cpu')
    model_nmtlr.fit(X_train_scaled.values.astype('float32'), y_train_nmtlr,
                batch_size=best_params_nmtlr['batch_size'],
                epochs=best_params_nmtlr['num_epochs'], verbose=False)
    models['NMTLR'] = model_nmtlr
    

#     param_grid = {
#     'max_depth': [3, 6, 10],
#     'eta': [0.1, 0.2, 0.3],
#     'colsample_bytree': [0.7, 0.8, 0.9],
#     'subsample': [0.7, 0.8, 0.9],
#     'n_estimators': [100, 200],
# }
#     xgboost_cv(X_train_scaled, y_train_df, param_grid)
  
    # # # 假设 X_train_scaled 和 y_train 包含了训练数据
    # dtrain = xgb.DMatrix(X_train_scaled, label=y_train_df['time'])
    # model_xgb = xgb.train(best_params_xgb, dtrain, num_boost_round=200)
    # models['XGBoost'] = model_xgb

    # from IPython import embed;embed()
    model_svm = FastSurvivalSVM()
    model_svm.fit(X_train_scaled, y_train)
    models['svm'] = model_svm

    # # 初始化 Weibull AFT 模型
    # aft_model = WeibullAFTFitter(penalizer=0.05, l1_ratio=0.1)
    # train_data = pd.concat([X_train.reset_index(drop=True),
    #                 pd.DataFrame({'time': y_train['time'], 'status': y_train['event']})], axis=1)
    # aft_model.fit(train_data, duration_col='time', event_col='status')
    # models['AFT'] = aft_model

    return models