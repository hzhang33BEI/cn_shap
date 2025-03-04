import itertools
from lifelines import CoxPHFitter
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from tqdm import tqdm
import logging
import concurrent.futures
import os
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sksurv.metrics import concordance_index_censored
import torch
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import xgboost as xgb
from models import train_base_models
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
from sklearn.preprocessing import StandardScaler


# 定义模型评估函数（返回特征组合和C-index）
def evaluate_model(features):
    # try:
    train_data = pd.concat([X_train[features].reset_index(drop=True),
                            y_train_df.reset_index(drop=True)], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_data, duration_col='time', event_col='status')
    # cph_predict = cph.predict_partial_hazard(X_train[features])
    c_index = cph.score(train_data, 
                            scoring_method="concordance_index")
    return (features, c_index)
    # except Exception as e:
    #     return (features, -1)  # 返回无效值表示失败


def evaluate_models(model, name, X, y):
    # 计算 C-index
    if name == 'CoxPH':
        x_df = X.reset_index(drop=True)
        y_df = pd.DataFrame({'time': y['time'], 'status': y['event']})
        c_index = model.score(pd.concat([x_df, y_df], axis=1),
                              scoring_method="concordance_index")
    elif name == 'AFT':
        x_df = X.reset_index(drop=True)
        y_df = pd.DataFrame({'time': y['time'], 'status': y['event']})
        # 选择评估时间点 t，通常可以用测试集中观察到事件的时间中位数
        event_times_test = y['time'][y['event'] == 1]
        t = np.median(event_times_test) if len(event_times_test) > 0 else np.median(y['time'])
        
        # 利用 AFT 模型预测生存函数，返回的 surv_funcs 为一个 DataFrame，
        # 行索引为时间点，列为各个样本的生存概率
        surv_funcs = model.predict_survival_function(x_df)
        available_times = np.array(surv_funcs.index, dtype=float)
        pos = np.searchsorted(available_times, t, side='right') - 1
        
        if pos < 0:
            surv_prob = surv_funcs.iloc[0].values
        elif pos >= len(available_times):
            surv_prob = surv_funcs.iloc[-1].values
        else:
            surv_prob = surv_funcs.iloc[pos].values

        from lifelines.utils import concordance_index
        c_index = concordance_index(y_df['time'], surv_prob, y_df['status'])

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
        elif name in ['XGBoost']:
            dX = xgb.DMatrix(X)
            pred = model.predict(dX)
        else:
            pred = model.predict(X)
        c_index = concordance_index_censored(y['event'], y['time'], pred)[0]
    
    return c_index


def train_base_models(model_name, X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, y_train_df, y_test_df):
    models = {}
    if model_name == 'CoxPH':
        # CoxPH模型
        train_data = pd.concat([X_train.reset_index(drop=True),
                            pd.DataFrame({'time': y_train['time'], 'status': y_train['event']})], axis=1)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_data, duration_col='time', event_col='status')
        models['CoxPH'] = cph

    elif model_name == 'RSF':
    # 随机生存森林
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, 
                                min_samples_leaf=10, n_jobs=-1, random_state=seed)
        rsf.fit(X_train_scaled, y_train)
        models['RSF'] = rsf
    
    elif model_name == 'GBSA':
        # 梯度提升生存分析
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

    # print("开始 DeepSurv 超参数调优……")
    # best_params_ds, best_score_ds, ds_scores = deep_surv_cv(X_train_scaled, y_train_df, param_grid_ds)
    # print("DeepSurv 最佳超参数：", best_params_ds, "对应 C-index：", best_score_ds)

    elif model_name == 'DeepSurv':
        # 用最佳超参数在全训练集上重新训练 DeepSurv 模型
        ds_model = LinearMultiTaskModel()
        ds_model.fit(X_train_scaled, y_train['time'], y_train['event'], 
                    init_method='orthogonal', optimizer=best_params_ds['optimizer'],
                    lr=best_params_ds['lr'], num_epochs=best_params_ds['num_epochs'], verbose=False)
        models['DeepSurv'] = ds_model



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
    elif model_name == 'DeepHit':
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

    elif model_name == 'NMTLR':
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
    elif model_name == 'svm':
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

def preprocess_data(data, features):
    select_cols = features + ['time', 'Status']
    data = data[select_cols]
    X = data.drop(['time', 'Status'], axis=1)
    y = data[['time', 'Status']].rename(columns={'time': 'time', 'Status': 'status'})
    
    X_train, X_test, y_train_df, y_test_df = train_test_split(X, y, test_size=0.3, random_state=seed)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    y_train = Surv.from_arrays(event=y_train_df['status'], time=y_train_df['time'])
    y_test = Surv.from_arrays(event=y_test_df['status'], time=y_test_df['time'])
    
    return X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, y_train_df, y_test_df,scaler


if __name__ == '__main__':

    data_path = './data/augmented_12.1.csv'
    data = pd.read_csv(data_path)
    # data = data.drop(['Regional LN surgery', 'Distant metastasis'], axis=1) #exceeed 35% missing data
    data.BMI = data.W * 100 * 100 / data.H / data.H
    data.ALT_AST = data.ALT / data.AST
    # 生成所有特征组合
    all_columns = list(data.columns)
    all_columns.remove('Status')
    all_columns.remove('time')
    
    from IPython import embed;embed()
    exit()
    feature_subsets = []
    for n in range(1, len(all_columns) + 1):
        for subset in itertools.combinations(all_columns, n):
            if len(list(subset)) < 5:
                continue
            feature_subsets.append(list(subset))
    
    total_combinations = len(feature_subsets)

    model_names = ["CoxPH", "RSF", "GBSA", "DeepSurv", "DeepHit", "NMTLR", "svm"]
    for model_name in model_names:
        print("{} processing {} {}".format("*"*10, model_name, "*"*10))
        # 初始化最佳值
        best_c_index = 0
        best_features = None

        # 配置日志
        logger = logging.getLogger(model_name)  # 为每个模型创建一个独立的 logger
        logger.setLevel(logging.INFO)  # 设置日志级别

        # 创建文件 handler，每个模型一个日志文件
        file_handler = logging.FileHandler('logs/{}_feature_selection.log'.format(model_name), mode='w')
        file_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将 handler 添加到 logger
        logger.addHandler(file_handler)
        

        with tqdm(total=total_combinations, desc="Searching for best feature combination") as pbar:
            for n in range(1, len(all_columns) + 1)[::-1]:
                for feature_subset in itertools.combinations(all_columns, n):
                    current_features = feature_subset = list(feature_subset)
                    if len(current_features) < 23:
                        continue
                    from IPython import embed;embed()
                    exit(0)
                    X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, y_train_df, y_test_df,scaler = preprocess_data(data=data, features=current_features)
                    current_model = train_base_models(model_name, X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, y_train_df, y_test_df)
                    if model_name in ['CoxPH']:
                        # 对于CoxPH和DeepSurv，直接使用X_test
                        current_c_index = evaluate_models(current_model[model_name], model_name, X_test, y_test)
                    else:
                        # 其他模型使用X_test_scaled
                        current_c_index = evaluate_models(current_model[model_name], model_name, X_test_scaled, y_test)
                    # 记录每个组合的评估结果到日志文件
                    logger.info(f"Evaluating {current_features}: C-index = {current_c_index}")
                    
                    # 如果找到更好的C-index，则更新
                    if current_c_index > best_c_index:
                        best_c_index = current_c_index
                        best_features = feature_subset
                        logger.info(f"New best C-index: {best_c_index} with features {best_features}")
                    
                    pbar.update(1)  # 更新进度条

        # 输出最终结果
        logger.info(f"Best C-index: {best_c_index}")
        logger.info(f"Best feature combination: {best_features}")