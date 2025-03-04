import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sksurv.metrics import concordance_index_censored
import torch
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import xgboost as xgb

# 4. 模型评估
def evaluate_model(model, name, X, y):
    results = {}
    
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
    results['C-index'] = c_index
    return results
    """
    # 新增部分：计算AUC、ROC和校准度
    # 选择评估时间点t（使用测试集事件时间的中位数）
    event_times_test = y['time'][y['event']]
    t = np.median(event_times_test) if len(event_times_test) > 0 else np.median(y['time'])
    
    # 获取每个样本在时间t的事件概率
    event_prob = []
    if name in ['CoxPH']:
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
            # from IPython import embed;embed()
            event_prob = 1 - surv_funcs[:, pos]
            # event_prob = surv_funcs[pos].y
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
    elif name in ['XGBoost']:
        dX = xgb.DMatrix(X)
        raw_preds = model.predict(dX)
        event_prob = 1 / (1 + np.exp(-raw_preds))  # Sigmoid转化
    elif name in ['svm']:
        risk_scores = model.predict(X)
        event_prob = 1 / (1 + np.exp(-risk_scores))  # 使用逻辑函数转换
    elif name in ['AFT']:
        # 对于 AFT 模型，使用 predict_survival_function 得到每个样本的生存函数
        surv_funcs = model.predict_survival_function(X)
        # surv_funcs 是一个 DataFrame，其 index 为时间点，列为各个样本的生存函数值
        available_times = np.array(surv_funcs.index, dtype=float)
        pos = np.searchsorted(available_times, t, side='right') - 1
        if pos < 0:
            surv_prob = surv_funcs.iloc[0].values
        elif pos >= len(available_times):
            surv_prob = surv_funcs.iloc[-1].values
        else:
            surv_prob = surv_funcs.iloc[pos].values
        event_prob = 1 - surv_prob

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
    """
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
