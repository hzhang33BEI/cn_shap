import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from lifelines import CoxPHFitter, WeibullAFTFitter
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
from pysurvival.models.multi_task import LinearMultiTaskModel

import torch
import torchtuples as tt
from pycox.models import DeepHitSingle, mtlr
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.calibration import calibration_curve

# 设置全局随机种子
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

# 数据预处理（保持原样）
columns = [
    'Race', 'Marital', 'Year', 'Age', 'Gender', 'Histological type', 
    'Primary site', 'Stage', 'Grade', 'Surgery', 'Radiotherapy', 
    'Chemotherapy', 'Tumor size', 'Number of tumors', 'Tumor extension',
    'Distant metastasis', 'Survival months', 'Status', 'Laterality',
    'Regional LN surgery', 'Systemic therapy', 'Metastasis to bone',
    'T stage', 'N stage', 'M stage', 'Metastasis to brain/liver/lung'
]
data = pd.read_csv('data/data_cox1.csv', usecols=columns)

# select_cols = ['Age', 'Race', 'Marital', 'Gender', 'Histological type', 'Primary site', 
#                'Grade',  'Radiotherapy', 'Chemotherapy', 'Tumor size', 'Distant metastasis', 
#                'Metastasis to bone', 'M stage', 'Survival months', 'Status']
select_cols = [
    'Race', 'Marital',  'Age', 'Gender', 'Histological type', 
    'Primary site', 'Grade', 'Surgery', 'Radiotherapy', 
    'Chemotherapy', 'Tumor size', 'Number of tumors', 
    'Distant metastasis', 'Survival months', 'Status', 
    'M stage'
]
data = data[select_cols]

X = data.drop(['Survival months', 'Status'], axis=1)
y = data[['Survival months', 'Status']].rename(columns={'Survival months': 'time', 'Status': 'status'})

X_train, X_test, y_train_df, y_test_df = train_test_split(X, y, test_size=0.3, random_state=seed)
# from IPython import embed;embed()
# exit()
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

y_train = Surv.from_arrays(event=y_train_df['status'], time=y_train_df['time'])
y_test = Surv.from_arrays(event=y_test_df['status'], time=y_test_df['time'])

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

# 定义 DeepSurv 参数搜索网格
param_grid_ds = {
    'lr': [1e-3, 1e-4],
    'num_epochs': [200, 400],
    'optimizer': ['adam']
}

print("开始 DeepSurv 超参数调优……")
best_params_ds, best_score_ds, ds_scores = deep_surv_cv(X_train_scaled, y_train_df, param_grid_ds)
print("DeepSurv 最佳超参数：", best_params_ds, "对应 C-index：", best_score_ds)

# 用最佳超参数在全训练集上重新训练 DeepSurv 模型
ds_model = LinearMultiTaskModel()
ds_model.fit(X_train_scaled, y_train['time'], y_train['event'], 
             init_method='orthogonal', optimizer=best_params_ds['optimizer'],
             lr=best_params_ds['lr'], num_epochs=best_params_ds['num_epochs'], verbose=False)
models['DeepSurv'] = ds_model

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
best_params_deephit = {'lr': 0.0001, 'num_epochs': 200, 'batch_size': 256, 'num_nodes': [64, 64], 'dropout': 0.2, 'num_durations': 120} 
best_params_nmtlr = {'lr': 0.001, 'num_epochs': 200, 'batch_size': 512, 'num_nodes': [32, 32], 'dropout': 0.2, 'num_durations': 120}
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

# 4. 模型评估
def evaluate_model(model, name, X, y):
    results = {}
    
    # 计算 C-index
    if name == 'CoxPH':
        x_df = X.reset_index(drop=True)
        y_df = pd.DataFrame({'time': y['time'], 'status': y['event']})
        c_index = model.score(pd.concat([x_df, y_df], axis=1),
                              scoring_method="concordance_index")
    else:
        if name == 'DeepSurv':
            pred = model.predict_risk(X)
        elif name in ['DeepHit', 'NMTLR']:
            x_tensor = torch.tensor(X.values.astype('float32'))
            surv_df = model.predict_surv_df(x_tensor)
            t0 = np.median(y['time'])
            available_times = np.array(surv_df.index, dtype=float)
            closest_time = available_times[np.argmin(np.abs(available_times - t0))]
            pred = 1 - surv_df.loc[closest_time].values
        else:
            pred = model.predict(X)
        c_index = concordance_index_censored(y['event'], y['time'], pred)[0]
    results['C-index'] = c_index
    
    # 新增部分：计算AUC、ROC和校准度
    # 选择评估时间点t（使用测试集事件时间的中位数）
    event_times_test = y['time'][y['event']]
    t = np.median(event_times_test) if len(event_times_test) > 0 else np.median(y['time'])
    
    # 获取每个样本在时间t的事件概率
    event_prob = []
    if name == 'CoxPH':
        baseline_survival = model.baseline_survival_
        times_cox = baseline_survival.index.values
        # from IPython import embed;embed()
        # exit()
        pos = np.searchsorted(times_cox, t, side='right') - 1
        if pos < 0:
            s0_t = 1.0
        elif pos >= len(times_cox):
            s0_t = 0.0
        else:
            s0_t = baseline_survival.iloc[pos, 0]
        risk_scores = model.predict_partial_hazard(X)
        surv_prob = np.power(s0_t, np.exp(risk_scores))
        event_prob = 1 - surv_prob
    elif name == 'RSF':
        # RandomSurvivalForest 的 predict_survival_function 返回数组
        surv_funcs = model.predict_survival_function(X)
        times = model.event_times_
        pos = np.searchsorted(times, t, side='right') - 1
        if pos < 0:
            event_prob = 1 - surv_funcs[:, 0]
        elif pos >= len(times):
            event_prob = 1 - surv_funcs[:, -1]
        else:
            event_prob = 1 - surv_funcs[:, pos]
    elif name == 'GBSA':
        # GradientBoostingSurvivalAnalysis 返回风险分数
        risk_scores = model.predict(X)
        # 将风险分数转换为事件概率（假设风险分数越高，事件概率越高）
        event_prob = 1 / (1 + np.exp(-risk_scores))  # 使用逻辑函数转换
    elif name == 'DeepSurv':
        times_array = np.array([t])
        survival = model.predict_survival(X.values, times_array).flatten()
        event_prob = 1 - survival
    elif name in ['DeepHit', 'NMTLR']:
        x_tensor = torch.tensor(X.values.astype('float32'))
        surv_df = model.predict_surv_df(x_tensor)
        available_times = surv_df.index.values
        closest_idx = np.argmin(np.abs(available_times - t))
        event_prob = 1 - surv_df.iloc[closest_idx].values
    else:
        event_prob = model.predict(X)
    
    # 构建二元标签（排除在t之前删失的样本）
    mask = (y['time'] > t) | (y['event'] == 1)
    y_binary = ((y['time'] <= t) & y['event']).astype(int)
    y_binary_filtered = y_binary[mask]
    event_prob_filtered = np.array(event_prob)[mask]
    
    # 计算AUC和ROC曲线
    if len(np.unique(y_binary_filtered)) >= 2:
        fpr, tpr, _ = roc_curve(y_binary_filtered, event_prob_filtered)
        roc_auc = auc(fpr, tpr)
        results['AUC'] = roc_auc
        results['ROC'] = (fpr, tpr, roc_auc)
    else:
        results['AUC'] = np.nan
        results['ROC'] = (np.nan, np.nan, np.nan)
        print(f"Warning: Only one class present in {name}. AUC not computed.")
    
    # 计算校准曲线
    if len(np.unique(y_binary_filtered)) >= 2 and len(event_prob_filtered) > 0:
        prob_true, prob_pred = calibration_curve(y_binary_filtered, event_prob_filtered, n_bins=10)
        results['Calibration'] = (prob_true, prob_pred)
    else:
        results['Calibration'] = (np.nan, np.nan)
    
    return results

def evaluate_model_at_time(model, name, X, y, time_point):
    results = {}
    
    # 获取每个样本在指定时间点的事件概率
    event_prob = []
    if name == 'CoxPH':
        baseline_survival = model.baseline_survival_
        times_cox = baseline_survival.index.values
        pos = np.searchsorted(times_cox, time_point, side='right') - 1
        if pos < 0:
            s0_t = 1.0
        elif pos >= len(times_cox):
            s0_t = 0.0
        else:
            s0_t = baseline_survival.iloc[pos, 0]
        risk_scores = model.predict_partial_hazard(X)
        surv_prob = np.power(s0_t, np.exp(risk_scores))
        event_prob = 1 - surv_prob
    elif name == 'RSF':
        # RandomSurvivalForest 的 predict_survival_function 返回数组
        surv_funcs = model.predict_survival_function(X)
        times = model.event_times_
        pos = np.searchsorted(times, time_point, side='right') - 1
        available_times = model.event_times_  # RSF 模型保存的所有时间点（升序数组）
        if pos < 0:
            event_prob = 1 - surv_funcs[:, 0]
        elif pos >= len(times):
            event_prob = 1 - surv_funcs[:, -1]
        else:
            # event_prob = 1 - surv_funcs[:, pos]
            surv_prob = np.array([sf(available_times[pos]) if callable(sf) else sf[pos]
                        for sf in surv_funcs])
            event_prob = 1 - surv_prob
    elif name == 'GBSA':
        # GradientBoostingSurvivalAnalysis 返回风险分数
        risk_scores = model.predict(X)
        # 将风险分数转换为事件概率（假设风险分数越高，事件概率越高）
        event_prob = 1 / (1 + np.exp(-risk_scores))  # 使用逻辑函数转换
    elif name == 'DeepSurv':
        times_array = np.array([time_point])
        survival = model.predict_survival(X.values, times_array).flatten()
        event_prob = 1 - survival
        event_prob = np.clip(event_prob, 0, 1)  # 将超出范围的值裁剪到 [0, 1]
        # from IPython import embed;embed()
        # exit()
    elif name in ['DeepHit', 'NMTLR']:    
        x_tensor = torch.tensor(X.values.astype('float32'))
        surv_df = model.predict_surv_df(x_tensor)
        available_times = surv_df.index.values
        # 寻找最后一个不超过time_point的时间点
        closest_idx = np.searchsorted(available_times, time_point, side='right') - 1
        if closest_idx < 0:
            closest_idx = 0
        elif closest_idx >= len(available_times):
            closest_idx = len(available_times) - 1
        event_prob = 1 - surv_df.iloc[closest_idx].values
    else:
        event_prob = model.predict(X)
    
    # 构建二元标签（排除在指定时间点之前删失的样本）
    mask = (y['time'] > time_point) | (y['event'] == 1)
    y_binary = ((y['time'] <= time_point) & y['event']).astype(int)
    y_binary_filtered = y_binary[mask]
    event_prob_filtered = np.array(event_prob)[mask]
    
    # 计算AUC和ROC曲线
    if len(np.unique(y_binary_filtered)) >= 2:
        fpr, tpr, _ = roc_curve(y_binary_filtered, event_prob_filtered)
        roc_auc = auc(fpr, tpr)
        results['AUC'] = roc_auc
        results['ROC'] = (fpr, tpr, roc_auc)
    else:
        results['AUC'] = np.nan
        results['ROC'] = (np.nan, np.nan, np.nan)
        print(f"Warning: Only one class present in {name} at {time_point}-year. AUC not computed.")
    
    # 计算校准曲线
    if len(np.unique(y_binary_filtered)) >= 2 and len(event_prob_filtered) > 0:
        prob_true, prob_pred = calibration_curve(y_binary_filtered, event_prob_filtered, n_bins=10)
        results['Calibration'] = (prob_true, prob_pred)
    else:
        results['Calibration'] = (np.nan, np.nan)
    
    return results

# 定义评估的时间点
time_points = [1, 3, 5, 10]   # 1-year, 3-year, 5-year, 10-year
time_points = [t * 12 for t in time_points]

# 评估所有模型并收集结果
all_results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    all_results[name] = {}
    for time_point in time_points:
        print(f"  Time point: {time_point}-year")
        if name in ['CoxPH']:
            res = evaluate_model_at_time(model, name, X_test, y_test, time_point)
        else:
            res = evaluate_model_at_time(model, name, X_test_scaled, y_test, time_point)
        all_results[name][time_point] = res
        print(f"    C-index: {res.get('C-index', 'N/A')}, AUC: {res.get('AUC', 'N/A')}")


# 绘制所有模型的 ROC 曲线（按时间点）
for time_point in time_points:
    plt.figure(figsize=(8, 6))
    for name, res in all_results.items():
        if time_point in res and 'ROC' in res[time_point] and not np.isnan(res[time_point]['ROC'][0]).any():
            fpr, tpr, roc_auc = res[time_point]['ROC']
            # from IPython import embed;embed()
            # exit()
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for All Models at {time_point / 12}-year')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_all_models_{time_point / 12}year.png')
    plt.close()


# 绘制所有模型的校准曲线（按时间点）
for time_point in time_points:
    plt.figure(figsize=(8, 6))
    for name, res in all_results.items():
        if time_point in res and 'Calibration' in res[time_point] and not np.isnan(res[time_point]['Calibration'][0]).any():
            prob_true, prob_pred = res[time_point]['Calibration']
            plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(f'Calibration Curves for All Models at {time_point / 12}-year')
    plt.legend()
    plt.savefig(f'calibration_all_models_{time_point / 12}year.png')
    plt.close()


# 评估所有模型
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    if name in ['CoxPH']:
        # 对于CoxPH和DeepSurv，直接使用X_test
        res = evaluate_model(model, name, X_test, y_test)
    else:
        # 其他模型使用X_test_scaled
        res = evaluate_model(model, name, X_test_scaled, y_test)
    results[name] = res
    print(f"{name} C-index: {res['C-index']:.4f}, AUC: {res.get('AUC', np.nan):.4f}")

print("\nFinal Evaluation Results:")
for name, res in results.items():
    print(f"{name}:")
    print(f"  C-index: {res['C-index']:.4f}")
    print(f"  AUC: {res.get('AUC', 'N/A'):.4f}")
    if 'Calibration' in res:
        print(f"  Calibration: {res['Calibration'][0].mean():.4f}")

# # 5. 模型可解释性（以 RSF 为例，通过 KernelExplainer）
# if 'RSF' in models:
#     def rsf_predict(X):
#         X_df = pd.DataFrame(X, columns=X_train_scaled.columns)
#         return models['RSF'].predict(X_df)
    
#     background = X_train_scaled.iloc[:50].values
#     explainer = shap.KernelExplainer(rsf_predict, background)
#     shap_values = explainer.shap_values(X_train_scaled.iloc[:10].values)
#     shap.summary_plot(shap_values, X_train_scaled.iloc[:10], feature_names=X_train_scaled.columns)

# ... [保留原有代码，直到模型训练完成] ...


# exit()


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

# 生成元特征
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
print("Generating meta-features...")
# from IPython import embed;embed()
# exit()

import itertools

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
    super_learner_models = dict(zip(candidate_model_name, candidate_model))
    print(candidate_model_name, "*"*20)
    meta_train = generate_meta_features(super_learner_models, X_train, y_train_df, scaler, kf)

    # 训练元模型（CoxPH）
    print("Training Super Learner...")
    meta_data = pd.concat([meta_train, y_train_df.reset_index(drop=True)], axis=1)
    super_learner = CoxPHFitter(penalizer=0.1)
    super_learner.fit(meta_data, 'time', 'status')

    # 生成测试集元特征
    print("Generating test meta-features...")
    meta_test = pd.DataFrame()
    for name in models:
        if name == 'CoxPH':
            pred = models[name].predict_partial_hazard(X_test)
        else:
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            if name == 'RSF':
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
    c_index = concordance_index_censored(y_test['event'], y_test['time'], super_pred)[0]
    print(f"Super Learner C-index: {c_index:.4f}")

    # # 将结果添加到metrics
    # results['SuperLearner'] = {'C-index': c_index}
    # print("评估指标：", results)