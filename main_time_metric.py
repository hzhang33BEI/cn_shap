from datasets import load_data, preprocess_data
from models import train_base_models
from model_evaluation import evaluate_model, evaluate_model_at_time
from matplotlib import pyplot as plt
import numpy as np
from super_learner import models_super_learner
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator
import shap
import torch
import xgboost as xgb

def plot_time_dependent_results(name, model, X, y, y_train_struct):
    y_test_struct = np.array([(e, t) for e, t in zip(y['event'], y['time'])],
                        dtype=[('event', bool), ('time', float)])
        # 通用配置
    EVAL_TIMES = np.quantile(y['time'][y['event']], [0.25, 0.5, 0.75])  # 评估时间点
    CALIB_BINS = 10  # 校准曲线分箱数

    """绘制时间相关曲线的主函数"""
    # 获取预测结果
    if name == 'DeepSurv':
        pred_risk = model.predict_risk(X)
    elif name in ['DeepHit', 'NMTLR']:
        x_tensor = torch.tensor(X.values.astype('float32'))
        surv_df = model.predict_surv_df(x_tensor)
        model_times = surv_df.index.values
        surv_funcs = surv_df.values.T
    elif name == 'GBSA':
        # # from IPython import embed;embed()
        # # surv_funcs = model.predict_survival_function(X, return_array=True)
        # # model_times = model.event_times_
        # risk_scores = surv_funcs = model.predict(X)
        # 从训练数据中提取唯一的事件时间点
        model_times = np.unique(y_train['time'][y_train['event']])
        surv_funcs = np.array([fn(model_times) for fn in model.predict_survival_function(X)])
    elif name == 'XGBoost':
        dX = xgb.DMatrix(X)
        pred_risk = model.predict(dX)
    
    # ====================
    # 时间相关ROC曲线
    # ====================
    if name in ['DeepHit', 'NMTLR', 'GBSA']:
        # 插值生存函数到评估时间点
        interp_surv = interp1d(model_times, surv_funcs, axis=1,
                              bounds_error=False, fill_value=(1.0, 0.0))(EVAL_TIMES)
        risk_scores = 1 - interp_surv
        
        # 计算每个时间点的AUC
        roc_aucs = [cumulative_dynamic_auc(y_train_struct, y_test_struct, 
                                          risk_scores[:,i], times=[t])[0][0]
                   for i, t in enumerate(EVAL_TIMES)]
    else:
        # 使用单一风险分数计算动态AUC
        roc_aucs = [cumulative_dynamic_auc(y_train_struct, y_test_struct,
                                          pred_risk, times=[t])[0][0]
                   for t in EVAL_TIMES]
    
    # 绘制ROC曲线
    plt.figure(figsize=(10,6))
    plt.plot(EVAL_TIMES, roc_aucs, 'bo-')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(f'{name} Time-dependent ROC Curve', fontsize=14)
    plt.grid(True)
    plt.show()

    # ====================
    # 时间相关校准曲线
    # ====================
    plt.figure(figsize=(10,6))
    
    if name in ['DeepHit', 'NMTLR', 'GBSA']:
        for i, t in enumerate(EVAL_TIMES):
            pred_risk_t = risk_scores[:,i]
            quantiles = np.quantile(pred_risk_t, np.linspace(0, 1, CALIB_BINS+1))
            groups = np.digitize(pred_risk_t, quantiles[:-1])
            
            bin_means, event_rates = [], []
            for g in range(1, CALIB_BINS+1):
                mask = (groups == g)
                if mask.sum() == 0: continue
                
                # 预测风险均值
                bin_means.append(pred_risk_t[mask].mean())
                
                # 实际事件率
                km_surv, km_times = kaplan_meier_estimator(y['event'][mask], y['time'][mask])
                actual_risk = 1 - km_surv[km_times >= t][0] if any(km_times >= t) else 1.0
                event_rates.append(actual_risk)
            
            plt.scatter(bin_means, event_rates, label=f't={t:.2f}')
    else:
        # 单一时间点（中位时间）校准
        t = np.median(y['time'])
        quantiles = np.quantile(pred_risk, np.linspace(0, 1, CALIB_BINS+1))
        groups = np.digitize(pred_risk, quantiles[:-1])
        
        bin_means, event_rates = [], []
        for g in range(1, CALIB_BINS+1):
            mask = (groups == g)
            if mask.sum() == 0: continue
            
            bin_means.append(pred_risk[mask].mean())
            km_surv, km_times = kaplan_meier_estimator(y['event'][mask], y['time'][mask])
            actual_risk = 1 - km_surv[km_times >= t][0] if any(km_times >= t) else 1.0
            event_rates.append(actual_risk)
        
        plt.scatter(bin_means, event_rates, label=f't={t:.2f}')
    
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('Predicted Risk', fontsize=12)
    plt.ylabel('Observed Risk', fontsize=12)
    plt.title(f'{name} Calibration Curve', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    # ====================
    # 特征重要性
    # ====================
    plt.figure(figsize=(10,6))
    
    # 原生特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plt.barh(range(len(importances)), importances, align='center')
        plt.yticks(range(len(X.columns)), X.columns)
        plt.title(f'{name} Feature Importance', fontsize=14)
    
    # SHAP特征重要性（适用于树模型）
    elif name in ['GBSA', 'XGBoost'] and X.shape[0] <= 1000:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title(f'{name} SHAP Feature Importance', fontsize=14)
    
    plt.xlabel('Importance Score', fontsize=12)
    plt.grid(True)
    plt.show()


    
if __name__ == "__main__":

    """
    Data processing
    """
    # data_path = 'data/data_cox1.csv'
    data_path = 'data/augmented_12.1.csv'
    data = load_data(data_path)
    X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, y_train_df, y_test_df,scaler = preprocess_data(data=data)

    """
    Model Training
    """
    models = train_base_models(X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, y_train_df, y_test_df)

    """
    Model Evalution
    """

    # for name, model in models.items():
    #     # ...原有评估代码...
    #     if not name in ['GBSA']:
    #         continue
    #     # 新增时间相关分析
    #     plot_time_dependent_results(name, model, X_test_scaled, y_test, y_train_struct=y_train)
        
    # 评估所有模型并收集结果
    time_points = [1, 2]   # 1-year, 3-year, 5-year, 10-year
    time_points = [t * 12 for t in time_points]
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

    """
    # 评估所有模型
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        if name in ['CoxPH', 'AFT']:
            # 对于CoxPH和DeepSurv，直接使用X_test
            res = evaluate_model(model, name, X_test, y_test)
        else:
            # 其他模型使用X_test_scaled
            res = evaluate_model(model, name, X_test_scaled, y_test)
        results[name] = res
        print(f"{name} C-index: {res['C-index']:.4f}, AUC: {res.get('AUC', np.nan):.4f}")

    # Super learner
    models_super_learner(models=models, X_train=X_train, y_train_df=y_train_df, X_test=X_test, y_test=y_test, scaler=scaler)

    """
