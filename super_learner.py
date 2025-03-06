
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from pysurvival.models.multi_task import LinearMultiTaskModel
from lifelines import CoxPHFitter, WeibullAFTFitter
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle, mtlr
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from config import SEED, best_params_deephit, best_params_nmtlr, best_params_ds


# 定义生成交叉验证元特征的函数
def generate_meta_features(models, X_train, y_train_df, scaler, cv):
    n_samples = X_train.shape[0]
    n_models = len(models)
    meta_features = np.zeros((n_samples, n_models))
    model_names = list(models.keys())
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        print(f"Processing Fold {fold+1}")
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train_df.iloc[train_idx]
        
        for model_idx, name in enumerate(model_names):
            print(f"  Training {name}")
            # 根据模型类型处理数据
            if name == 'CoxPH':
                # 训练CoxPH
                train_data = pd.concat([X_tr, y_tr], axis=1)
                model = CoxPHFitter(penalizer=0.1)
                model.fit(train_data, 'time', 'status')
                pred = model.predict_partial_hazard(X_val)
            else:
                # 标准化数据
                X_tr_scaled = scaler.transform(X_tr)
                X_tr_scaled = pd.DataFrame(X_tr_scaled, columns=X_tr.columns)
                X_val_scaled = scaler.transform(X_val)
                X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
                
                # 实例化新模型（避免数据泄漏）
                if name == 'RSF':
                    model = RandomSurvivalForest(**models[name].get_params())
                    y_tr_struct = Surv.from_arrays(y_tr['status'], y_tr['time'])
                    model.fit(X_tr_scaled, y_tr_struct)
                    pred = model.predict(X_val_scaled)
                elif name == 'GBSA':
                    model = GradientBoostingSurvivalAnalysis(**models[name].get_params())
                    y_tr_struct = Surv.from_arrays(y_tr['status'], y_tr['time'])
                    model.fit(X_tr_scaled, y_tr_struct)
                    pred = model.predict(X_val_scaled)
                elif name == 'svm':
                    from sksurv.svm import FastSurvivalSVM
                    model = FastSurvivalSVM()
                    y_tr_struct = Surv.from_arrays(y_tr['status'], y_tr['time'])
                    model.fit(X_tr_scaled, y_tr_struct)
                    pred = model.predict(X_val_scaled)
                elif name == 'DeepSurv':
                    model = LinearMultiTaskModel()
                    model.fit(X_tr_scaled, y_tr['time'].values, y_tr['status'].values,
                             init_method='orthogonal',
                             optimizer=best_params_ds['optimizer'],
                             lr=best_params_ds['lr'],
                             num_epochs=best_params_ds['num_epochs'],
                             verbose=False)
                    pred = model.predict_risk(X_val_scaled)
                elif name == 'DeepHit':
                    labtrans = LabTransDiscreteTime(best_params_deephit['num_durations'])
                    durations = y_tr['time'].values
                    events = y_tr['status'].values.astype(bool)
                    y_lab = labtrans.fit_transform(durations, events)
                    
                    net = tt.practical.MLPVanilla(
                        X_tr_scaled.shape[1],
                        best_params_deephit['num_nodes'],
                        labtrans.out_features,
                        batch_norm=True,
                        dropout=best_params_deephit['dropout']
                    )
                    optimizer = tt.optim.Adam(lr=best_params_deephit['lr'])
                    model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
                    model.set_device('cpu')
                    model.fit(
                        X_tr_scaled.values.astype('float32'), y_lab,
                        batch_size=best_params_deephit['batch_size'],
                        epochs=best_params_deephit['num_epochs'],
                        verbose=False
                    )
                    surv = model.predict_surv_df(X_val_scaled.values.astype('float32'))
                    t_median = np.median(y_tr['time'])
                    times = np.array(surv.index)
                    idx = np.argmin(np.abs(times - t_median))
                    pred = 1 - surv.iloc[idx].values
                
                elif name == 'NMTLR':
                    # 数据标准化（使用当前折的scaler）
                    fold_scaler = StandardScaler()
                    X_tr_scaled = fold_scaler.fit_transform(X_tr)
                    X_tr_scaled = pd.DataFrame(X_tr_scaled, columns=X_tr.columns)
                    X_val_scaled = fold_scaler.transform(X_val)
                    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
                    
                    # 转换生存时间
                    labtrans = LabTransDiscreteTime(best_params_nmtlr['num_durations'])
                    durations = y_tr['time'].values
                    events = y_tr['status'].values.astype(bool)
                    y_lab = labtrans.fit_transform(durations, events)
                    
                    # 构建网络
                    net = tt.practical.MLPVanilla(
                        X_tr_scaled.shape[1],
                        best_params_nmtlr['num_nodes'],
                        labtrans.out_features,
                        batch_norm=True,
                        dropout=best_params_nmtlr['dropout']
                    )
                    
                    # 初始化模型
                    optimizer = tt.optim.Adam(lr=best_params_nmtlr['lr'])
                    model = mtlr.MTLR(net, optimizer, duration_index=labtrans.cuts)
                    model.set_device('cpu')
                    
                    # 训练模型
                    model.fit(
                        X_tr_scaled.values.astype('float32'), 
                        y_lab,
                        batch_size=best_params_nmtlr['batch_size'],
                        epochs=best_params_nmtlr['num_epochs'],
                        verbose=False
                    )
                    
                    # 生成预测
                    x_tensor = torch.tensor(X_val_scaled.values.astype('float32'))
                    surv_df = model.predict_surv_df(x_tensor)
                    
                    # 选择中位时间点
                    t_median = np.median(y_tr['time'])
                    available_times = np.array(surv_df.index, dtype=float)
                    closest_time = available_times[np.argmin(np.abs(available_times - t_median))]
                    
                    # 获取预测风险值并归一化
                    pred = 1 - surv_df.loc[closest_time].values
                    pred = (pred - pred.min()) / (pred.max() - pred.min())
                
            # 归一化预测值
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            meta_features[val_idx, model_idx] = pred
            
    return pd.DataFrame(meta_features, columns=model_names)


import logging
import itertools

def models_super_learner(models, X_train, y_train_df, X_test, y_test, scaler, y_train=None, external_X_test=None, external_y_test=None, external_scaler=None):
    # 生成元特征
    kf = KFold(n_splits=2, shuffle=True, random_state=SEED)
    print("Generating meta-features...")
    logging.info("Generating meta-features...")
    # from IPython import embed;embed()
    # exit()


    # 获取模型的所有名称
    model_names = list(models.keys())

    # 生成所有非空组合（至少选择一个模型）
    candidate_models = {}
    for r in range(1, len(model_names) + 1):
        for combo in itertools.combinations(model_names, r):
            # 为每个组合创建字典，组合名称为键，组合对应的模型列表为值
            candidate_models[combo] = [models[name] for name in combo]
    # from IPython import embed;embed()
    for candidate_model_name, candidate_model in candidate_models.items():
        if len(candidate_model_name) == 1:
            continue
        super_learner_models = dict(zip(candidate_model_name, candidate_model))
        # print(candidate_model_name, "*"*20)
        # logging.info("Super learner for : {} ".format(candidate_model_name))
        meta_train = generate_meta_features(super_learner_models, X_train, y_train_df, scaler, kf)

        # 训练元模型（CoxPH）
        # print("Training Super Learner...")
        meta_data = pd.concat([meta_train, y_train_df.reset_index(drop=True)], axis=1)
        super_learner = CoxPHFitter(penalizer=0.1)
        super_learner.fit(meta_data, 'time', 'status')

        # 生成测试集元特征
        # print("Generating test meta-features...")
        meta_test = pd.DataFrame()
        for name in models:
            if name == 'CoxPH':
                pred = models[name].predict_partial_hazard(X_test)
            else:
                X_test_scaled = scaler.transform(X_test)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
                if name == 'RSF':
                    pred = models[name].predict(X_test_scaled)
                elif name == 'svm':
                    pred = models[name].predict(X_test_scaled)
                elif name == 'GBSA':
                    pred = models[name].predict(X_test_scaled)
                elif name == 'DeepSurv':
                    pred = models[name].predict_risk(X_test_scaled)
                elif name == 'DeepHit':
                    x_tensor = torch.tensor(X_test_scaled.values.astype('float32'))
                    surv = models[name].predict_surv_df(x_tensor)
                    t_median = np.median(y_test['time'])
                    times = np.array(surv.index)
                    idx = np.argmin(np.abs(times - t_median))
                    pred = 1 - surv.iloc[idx].values
                # 在生成测试集元特征的循环中补充NMTLR部分
                elif name == 'NMTLR':
                    x_tensor = torch.tensor(X_test_scaled.values.astype('float32'))
                    surv = models[name].predict_surv_df(x_tensor)
                    t_median = np.median(y_test['time'])
                    times = np.array(surv.index)
                    idx = np.argmin(np.abs(times - t_median))
                    pred_raw = 1 - surv.iloc[idx].values
                    pred = (pred_raw - pred_raw.min()) / (pred_raw.max() - pred_raw.min())
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            meta_test[name] = pred

        # 评估Super Learner
        super_pred = super_learner.predict_partial_hazard(meta_test)
        # from IPython import embed;embed()
        val_c_index = concordance_index_censored(y_test['event'], y_test['time'], super_pred)[0]

        meta_test = pd.DataFrame()
        for name in models:
            if name == 'CoxPH':
                pred = models[name].predict_partial_hazard(X_train)
            else:
                X_train_scaled = scaler.transform(X_train)
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                if name == 'RSF':
                    pred = models[name].predict(X_train_scaled)
                elif name == 'svm':
                    pred = models[name].predict(X_train_scaled)
                elif name == 'GBSA':
                    pred = models[name].predict(X_train_scaled)
                elif name == 'DeepSurv':
                    pred = models[name].predict_risk(X_train_scaled)
                elif name == 'DeepHit':
                    x_tensor = torch.tensor(X_train_scaled.values.astype('float32'))
                    surv = models[name].predict_surv_df(x_tensor)
                    t_median = np.median(y_train_df['time'])
                    times = np.array(surv.index)
                    idx = np.argmin(np.abs(times - t_median))
                    pred = 1 - surv.iloc[idx].values
                # 在生成测试集元特征的循环中补充NMTLR部分
                elif name == 'NMTLR':
                    x_tensor = torch.tensor(X_train_scaled.values.astype('float32'))
                    surv = models[name].predict_surv_df(x_tensor)
                    t_median = np.median(y_train_df['time'])
                    times = np.array(surv.index)
                    idx = np.argmin(np.abs(times - t_median))
                    pred_raw = 1 - surv.iloc[idx].values
                    pred = (pred_raw - pred_raw.min()) / (pred_raw.max() - pred_raw.min())
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            meta_test[name] = pred

        # 评估Super Learner
        super_pred = super_learner.predict_partial_hazard(meta_test)
        # from IPython import embed;embed()
        train_c_index = concordance_index_censored(y_train['event'], y_train['time'], super_pred)[0]

        meta_test = pd.DataFrame()
        for name in models:
            if name == 'CoxPH':
                pred = models[name].predict_partial_hazard(external_X_test)
            else:
                external_X_test_scaled = external_scaler.transform(external_X_test)
                external_X_test_scaled = pd.DataFrame(external_X_test_scaled, columns=external_X_test.columns)
                if name == 'RSF':
                    pred = models[name].predict(external_X_test_scaled)
                elif name == 'svm':
                    pred = models[name].predict(external_X_test_scaled)
                elif name == 'GBSA':
                    pred = models[name].predict(external_X_test_scaled)
                elif name == 'DeepSurv':
                    pred = models[name].predict_risk(external_X_test_scaled)
                elif name == 'DeepHit':
                    x_tensor = torch.tensor(external_X_test_scaled.values.astype('float32'))
                    surv = models[name].predict_surv_df(x_tensor)
                    t_median = np.median(external_y_test['time'])
                    times = np.array(surv.index)
                    idx = np.argmin(np.abs(times - t_median))
                    pred = 1 - surv.iloc[idx].values
                # 在生成测试集元特征的循环中补充NMTLR部分
                elif name == 'NMTLR':
                    x_tensor = torch.tensor(external_X_test_scaled.values.astype('float32'))
                    surv = models[name].predict_surv_df(x_tensor)
                    t_median = np.median(external_y_test['time'])
                    times = np.array(surv.index)
                    idx = np.argmin(np.abs(times - t_median))
                    pred_raw = 1 - surv.iloc[idx].values
                    pred = (pred_raw - pred_raw.min()) / (pred_raw.max() - pred_raw.min())
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            meta_test[name] = pred

        # 评估Super Learner
        super_pred = super_learner.predict_partial_hazard(meta_test)
        # from IPython import embed;embed()
        external_val_c_index = concordance_index_censored(external_y_test['event'], external_y_test['time'], super_pred)[0]
        
        
        logging.info("Super Learner -> {}:  Train-C-index: {} Internal-Val-C-index: {} External-Val-C-index: {}".format(candidate_model_name, train_c_index, val_c_index, external_val_c_index))