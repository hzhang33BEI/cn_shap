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

import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from scipy.interpolate import interp1d

def plot_rfs_calibration(model,model_name, X_test, y_test, max_time=None, n_points=100):
    """
    绘制无复发生存概率校准曲线
    参数：
    - model: 训练好的生存分析模型
    - X_test: 测试集特征
    - y_test: 测试集生存数据 (包含time和event列)
    - max_time: 最大显示时间（默认为测试集最大时间）
    - n_points: 时间点采样数量
    """
    # 准备数据
    event_times = y_test['time']
    event_indicators = y_test['event']
    
    # 计算真实生存曲线（Kaplan-Meier）
    km_times, km_surv = kaplan_meier_estimator(
        event_indicators.astype(bool), 
        event_times
    )
    # from IPython import embed;embed()
    # 获取预测生存函数
    # if hasattr(model, "predict_survival_function"):
    if model_name == 'RSF':
        # 树模型（GBSA）的处理方式
        surv_funcs = model.predict_survival_function(X_test)
        pred_times = model.event_times_
        pred_surv = np.array([fn(pred_times) for fn in surv_funcs]).mean(axis=0)
    elif model_name == 'CoxPH':
        pred_times = model.baseline_survival_.index.values
        
        # 计算每个样本的风险评分
        risk_scores = model.predict_log_partial_hazard(X_test).values
        
        # 生成预测生存曲线（样本平均）
        baseline_surv = model.baseline_survival_.values.ravel()
        pred_surv = np.array([baseline_surv ** np.exp(score) for score in risk_scores])
        pred_surv = pred_surv.mean(axis=0)
    else:
        # 深度学习模型的生存函数预测
        x_tensor = torch.tensor(X_test.values.astype('float32'))
        surv_df = model.predict_surv_df(x_tensor)
        pred_times = surv_df.index.values
        pred_surv = surv_df.values.mean(axis=1)
    
    # 设置时间范围
    max_time = max_time or min(event_times.max(), pred_times.max())
    eval_times = np.linspace(0, max_time, n_points)
    
    # 插值真实生存曲线
    km_interp = interp1d(km_times, km_surv, kind='previous', 
                        bounds_error=False, fill_value=(1.0, 0.0))(eval_times)
    
    # 插值预测生存曲线
    pred_interp = interp1d(pred_times, pred_surv, kind='linear', 
                          bounds_error=False, fill_value=(1.0, 0.0))(eval_times)
    
    # 绘图设置
    plt.figure(figsize=(10, 6))
    plt.plot(eval_times, km_interp, 'b-', lw=2, label='Ground Truth')
    plt.plot(eval_times, pred_interp, 'r--', lw=2, label='Predicted')
    
    # 美化图形
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Recurrence-Free Survival Probability', fontsize=12)
    plt.title('Recurrence-Free Survival Calibration Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')
    plt.xlim(0, max_time)
    plt.ylim(-0.05, 1.05)
    
    # 添加时间刻度标记（示例：每年标记）
    years = np.arange(0, int(max_time/365)+1)
    plt.xticks(years*365, [f"{y} year" for y in years])
    
    plt.show()



if __name__ == "__main__":
    """
    Data processing
    """
    # data_path = 'data/data_cox1.csv'
    data_path = "data/augmented_12.1.csv"
    data = load_data(data_path)
    (
        X_train_scaled,
        X_train,
        y_train,
        X_test_scaled,
        X_test,
        y_test,
        y_train_df,
        y_test_df,
        scaler,
    ) = preprocess_data(data=data)

    """
    Model Training
    """
    models = train_base_models(
        X_train_scaled,
        X_train,
        y_train,
        X_test_scaled,
        X_test,
        y_test,
        y_train_df,
        y_test_df,
    )
    
    # 使用示例

    plot_rfs_calibration(models['svm'],'svm' ,X_test_scaled, y_test, max_time=12)

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
    # time_points = [1, 2]   # 1-year, 3-year, 5-year, 10-year
    # time_points = [t * 12 for t in time_points]
    time_points = [12, 24]
    all_results = {}
    for name, model in models.items():
        if not name in [
            "CoxPH","RSF",
            "svm",
        ]:
            continue
        print(f"Evaluating {name}...")
        all_results[name] = {}
        for time_point in time_points:
            print(f"  Time point: {time_point}-year")
            if name in ["CoxPH"]:
                res = evaluate_model_at_time(model, name, X_test, y_test, time_point)
            else:
                res = evaluate_model_at_time(
                    model, name, X_test_scaled, y_test, time_point
                )
                # res = evaluate_model_at_time(model, name, X_train_scaled, y_train, time_point)
            all_results[name][time_point] = res
            print(
                f"    C-index: {res.get('C-index', 'N/A')}, AUC: {res.get('AUC', 'N/A')}"
            )

    # 绘制所有模型的 ROC 曲线（按时间点）
    for time_point in time_points:
        plt.figure(figsize=(8, 6))
        for name, res in all_results.items():
            if (
                time_point in res
                and "ROC" in res[time_point]
                and not np.isnan(res[time_point]["ROC"][0]).any()
            ):
                fpr, tpr, roc_auc = res[time_point]["ROC"]
                # from IPython import embed;embed()
                # exit()
                plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves for All Models at {time_point / 12}-year")
        plt.legend(loc="lower right")
        plt.savefig(f"roc_all_models_{time_point / 12}year.png")
        plt.close()

    # 绘制所有模型的校准曲线（按时间点）
    for time_point in time_points:
        plt.figure(figsize=(8, 6))
        for name, res in all_results.items():
            if (
                time_point in res
                and "Calibration" in res[time_point]
                and not np.isnan(res[time_point]["Calibration"][0]).any()
            ):
                prob_true, prob_pred = res[time_point]["Calibration"]
                # from IPython import embed;embed()
                # exit()
                plt.plot(prob_pred, prob_true, marker="o", label=name)
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")
        plt.title(f"Calibration Curves for All Models at {time_point / 12}-year")
        plt.legend()
        plt.savefig(f"calibration_all_models_{time_point / 12}year.png")
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
