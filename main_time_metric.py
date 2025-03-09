from datasets import load_data, preprocess_data, preprocess_external_data
from models import train_base_models
from model_evaluation import evaluate_model, evaluate_model_at_time
from matplotlib import pyplot as plt
import numpy as np
from super_learner import models_super_learner, specific_super_learner_training, generate_meta_features, get_meta_data
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
import pandas as pd

def plot_calibration(models, X_test, y_test, max_time=None, n_points=100, super_learner_fit_models=None):

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
        # 设置时间范围
    max_time = max_time or min(event_times.max(), pred_times.max())
    eval_times = np.linspace(0, max_time, n_points)

    # 计算真实生存曲线（Kaplan-Meier）
    km_times, km_surv = kaplan_meier_estimator(
        event_indicators.astype(bool), 
        event_times
    )
    # 插值真实生存曲线
    km_interp = interp1d(km_times, km_surv, kind='previous', 
                        bounds_error=False, fill_value=(1.0, 0.0))(eval_times)

        # 绘图设置
    plt.figure(figsize=(10, 6))
    plt.plot(eval_times, km_interp, 'b-',lw=2, label='Actual survival probability')
    

    # #save result to csv
    save_data = pd.DataFrame({'eval_times': eval_times, 'actual_survival_probability': km_interp})
    
    # 美化图形
    plt.xlabel('Time (month)', fontsize=12)
    plt.ylabel('Recurrence-Free Survival Probability', fontsize=12)
    plt.title('Recurrence-Free Survival Calibration Curve', fontsize=14)
    plt.xlim(0, max_time)
    plt.ylim(-0.05, 1.05)

    for model_name, model in models.items():
        print('**', model_name)
        # if model_name in ['DeepSurv', 'DeepHit', 'NMTLR', 'svm', 'RSF']:
        #     continue

        if model_name == 'RSF':
            # surv_funcs 是形状为 (n_samples, n_times) 的数组
            surv_funcs = model.predict_survival_function(X_test)
            pred_surv = surv_funcs.mean(axis=0)  # 直接在时间维度取平均
            pred_times = model.event_times_
        elif model_name == 'CoxPH':
            pred_times = model.baseline_survival_.index.values
            # 计算每个样本的风险评分
            risk_scores = model.predict_log_partial_hazard(X_test).values
            
            # 生成预测生存曲线（样本平均）
            baseline_surv = model.baseline_survival_.values.ravel()
            pred_surv = np.array([baseline_surv ** np.exp(score) for score in risk_scores])
            pred_surv = pred_surv.mean(axis=0)
        elif model_name == 'super_learner':
            pred_times = model.baseline_survival_.index.values
            # X_meta = get_meta_data(super_learner_fit_models, X_test, y_test['time'].mean())
            X_meta = get_meta_data(super_learner_fit_models, X_test, t_median=10.21516891628409) #time from y_train
            # 计算每个样本的风险评分
            # from IPython import embed;embed()
            # exit()
            risk_scores = model.predict_log_partial_hazard(X_meta).values
            
            # 生成预测生存曲线（样本平均）
            baseline_surv = model.baseline_survival_.values.ravel()
            pred_surv = np.array([baseline_surv ** np.exp(score) for score in risk_scores])
            pred_surv = pred_surv.mean(axis=0)
        elif model_name in ['DeepHit', 'NMTLR']:
            # 深度学习模型处理
            x_tensor = torch.tensor(X_test.values.astype('float32'))
            surv_df = model.predict_surv_df(x_tensor)
            pred_times = surv_df.index.values
            pred_surv = surv_df.values.mean(axis=1)  
        elif model_name == 'svm':
            risk_scores = model.predict(X_test)
            # 估计生存概率
            from sksurv.linear_model.coxph import BreslowEstimator 
            breslow = BreslowEstimator().fit(risk_scores, y_test["event"], y_test["time"])
            pred_times = np.unique(y_test["time"])
            pred_times = pred_times[pred_times <= max_time]  # 限制到最大时间
            
            surv_funcs = breslow.get_survival_function(risk_scores)
            pred_surv = np.array([fn(pred_times) for fn in surv_funcs]).mean(axis=0)
            # from IPython import embed;embed()
            # exit()
        elif model_name == 'GBSA':
            risk_scores = model.predict(X_test)
            # 估计生存概率
            from sksurv.linear_model.coxph import BreslowEstimator 
            breslow = BreslowEstimator().fit(risk_scores, y_test["event"], y_test["time"])
            pred_times = np.unique(y_test["time"])
            pred_times = pred_times[pred_times <= max_time]  # 限制到最大时间
            
            surv_funcs = breslow.get_survival_function(risk_scores)
            pred_surv = np.array([fn(pred_times) for fn in surv_funcs]).mean(axis=0)
            # from IPython import embed;embed()
            # exit()
        else: 
            surv_funcs = model.predict_survival_function(X_test)
            pred_times = model.event_times_
            pred_surv = np.array([fn(pred_times) for fn in surv_funcs]).mean(axis=0)
        

        
        # 插值预测生存曲线
        pred_interp = interp1d(pred_times, pred_surv, kind='linear', 
                            bounds_error=False, fill_value=(1.0, 0.0))(eval_times)
        
        plt.plot(eval_times, pred_interp, lw=2, label='{}'.format(model_name))

        save_data[model_name] = pred_interp
        
    # 添加时间刻度标记（示例：每年标记）
    months = np.arange(0, max_time + 1, 12)
    plt.xticks(months, [f"{y} " for y in months])
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')
    plt.savefig('calibration_all.png')
    plt.show()
    # save_data.to_csv('calibration_external_val.csv', index=False)


def plot_rfs_calibration(model,model_name, X_test, y_test, max_time=None, n_points=100, super_learner_fit_models=None):
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
        # surv_funcs 是形状为 (n_samples, n_times) 的数组
        surv_funcs = model.predict_survival_function(X_test)
        pred_surv = surv_funcs.mean(axis=0)  # 直接在时间维度取平均
        pred_times = model.event_times_
    elif model_name == 'CoxPH':
        pred_times = model.baseline_survival_.index.values
        # 计算每个样本的风险评分
        risk_scores = model.predict_log_partial_hazard(X_test).values
        
        # 生成预测生存曲线（样本平均）
        baseline_surv = model.baseline_survival_.values.ravel()
        pred_surv = np.array([baseline_surv ** np.exp(score) for score in risk_scores])
        pred_surv = pred_surv.mean(axis=0)
    elif model_name == 'super_learner':
        pred_times = model.baseline_survival_.index.values
        # X_meta = get_meta_data(super_learner_fit_models, X_test, y_test['time'].mean())
        X_meta = get_meta_data(super_learner_fit_models, X_test, t_median=10.21516891628409) #time from y_train
        # 计算每个样本的风险评分
        # from IPython import embed;embed()
        # exit()
        risk_scores = model.predict_log_partial_hazard(X_meta).values
        
        # 生成预测生存曲线（样本平均）
        baseline_surv = model.baseline_survival_.values.ravel()
        pred_surv = np.array([baseline_surv ** np.exp(score) for score in risk_scores])
        pred_surv = pred_surv.mean(axis=0)
    elif model_name in ['DeepHit', 'NMTLR']:
        # 深度学习模型处理
        x_tensor = torch.tensor(X_test.values.astype('float32'))
        surv_df = model.predict_surv_df(x_tensor)
        pred_times = surv_df.index.values
        pred_surv = surv_df.values.mean(axis=1)  
    elif model_name == 'svm':
        risk_scores = model.predict(X_test)
        # 估计生存概率
        from sksurv.linear_model.coxph import BreslowEstimator 
        breslow = BreslowEstimator().fit(risk_scores, y_test["event"], y_test["time"])
        pred_times = np.unique(y_test["time"])
        pred_times = pred_times[pred_times <= max_time]  # 限制到最大时间
        
        surv_funcs = breslow.get_survival_function(risk_scores)
        pred_surv = np.array([fn(pred_times) for fn in surv_funcs]).mean(axis=0)
        # from IPython import embed;embed()
        # exit()
    else:  # GBSA等树模型
        surv_funcs = model.predict_survival_function(X_test)
        pred_times = model.event_times_
        pred_surv = np.array([fn(pred_times) for fn in surv_funcs]).mean(axis=0)
    
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
    plt.xlabel('Time (month)', fontsize=12)
    plt.ylabel('Recurrence-Free Survival Probability', fontsize=12)
    plt.title('Recurrence-Free Survival Calibration Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')
    plt.xlim(0, max_time)
    plt.ylim(-0.05, 1.05)
    
    # 添加时间刻度标记（示例：每年标记）
    months = np.arange(0, max_time + 1, 12)
    plt.xticks(months, [f"{y} " for y in months])
    plt.savefig('calibration_{}.png'.format(model_name))
    plt.show()


def time_dependent_feature_importance_curve(model, model_name,train_data, test_data, select_cols):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold
    from sksurv.util import Surv
    from sksurv.metrics import brier_score

    # 定义十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # 用于存储各特征在各折中得到的 Brier 分数差（扰动后 - 原始）
    feature_importance_results = {feature: [] for feature in select_cols}

    # # 十折交叉验证
    # for train_index, test_index in kf.split(df):
    #     train_data = df.iloc[train_index].reset_index(drop=True)
    #     test_data = df.iloc[test_index].reset_index(drop=True)
        
    # 构造特征数据
    X_train = train_data[select_cols]
    X_test = test_data[select_cols]
    # 构造生存数据（"status"为事件指示，"time"为生存时间）
    y_train = Surv.from_dataframe("status", "time", train_data)
    y_test = Surv.from_dataframe("status", "time", test_data)
    
    # 针对当前测试集，生成时间网格，使用 endpoint=False 确保最大值不被包含
    t_min = test_data["time"].min()
    t_max = test_data["time"].max()
    time_grid = np.linspace(t_min, t_max, 8, endpoint=False)
    
    # 预测测试集的生存函数
    if model_name == 'RSF':
        surv_funcs = model.predict_survival_function(X_test, return_array=True)
        # 判断返回值类型：如果是可调用对象，则直接计算；否则需要对默认时间网格进行插值
        if hasattr(surv_funcs[0], '__call__'):
            surv_probs = np.asarray([[fn(t) for t in time_grid] for fn in surv_funcs])
        else:
            # rsf.predict_survival_function 返回的是数组，其默认时间网格存储在 rsf.event_times_
            default_times = model.event_times_
            surv_probs = np.asarray([np.interp(time_grid, default_times, sample_surv)
                                        for sample_surv in surv_funcs])
    elif model_name in ['DeepHit', 'NMTLR']:

        x_tensor = torch.tensor(X_test.values.astype('float32'))
        surv_probs = model.predict_surv(x_tensor)
        surv_probs = np.array(surv_probs)
    elif model_name in ['svm', 'GBSA']:
        from lifelines import KaplanMeierFitter, NelsonAalenFitter

        # 假设已训练模型 `model`
        risk_scores = model.predict(X_test_scaled)  # 预测风险分数（越大，风险越高）

        # 需要 `y_train`（生存时间和事件），用它来计算基线风险
        times, events = y_train['time'], y_train['status']

        # 计算基线累积风险函数
        naf = NelsonAalenFitter()
        # naf.fit(times, event_observed=events)
        naf.fit(times, event_observed=events, timeline=time_grid) 
        baseline_hazard = naf.cumulative_hazard_

        # 计算生存函数 S(t | X)
        surv_probs = np.exp(-np.outer(np.exp(risk_scores), baseline_hazard.values.flatten()))
    elif model_name in ['super_learner']:
        from lifelines import KaplanMeierFitter, NelsonAalenFitter
        X_meta = get_meta_data(super_learner_fit_models, X_test, t_median=10.21516891628409) #time from y_train
        # 计算每个样本的风险评分
        risk_scores = model.predict_log_partial_hazard(X_meta).values
        # 需要 `y_train`（生存时间和事件），用它来计算基线风险
        times, events = y_train['time'], y_train['status']

        # 计算基线累积风险函数
        naf = NelsonAalenFitter()
        # naf.fit(times, event_observed=events)
        naf.fit(times, event_observed=events, timeline=time_grid) 
        baseline_hazard = naf.cumulative_hazard_
        # 计算生存函数 S(t | X)
        surv_probs = np.exp(-np.outer(np.exp(risk_scores), baseline_hazard.values.flatten()))

    # 计算基线 Brier 分数
    times, baseline_scores = brier_score(y_train, y_test, surv_probs, time_grid)
    
    save_data = pd.DataFrame()

    # 对每个特征进行扰动，并计算扰动后的 Brier 分数变化
    for feature in select_cols:
        X_test_permuted = X_test.copy()
        X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature].values)
        
        if model_name == 'RSF':
            surv_funcs_perm = model.predict_survival_function(X_test_permuted)
            if hasattr(surv_funcs_perm[0], '__call__'):
                surv_probs_perm = np.asarray([[fn(t) for t in time_grid] for fn in surv_funcs_perm])
            else:
                surv_probs_perm = np.asarray([np.interp(time_grid, model.event_times_, sample_surv)
                                                for sample_surv in surv_funcs_perm])
        
        elif model_name in ['DeepHit','NMTLR']:
            x_tensor = torch.tensor(X_test_permuted.values.astype('float32'))
            surv_probs_perm = model.predict_surv(x_tensor)
            surv_probs_perm = np.array(surv_probs_perm)
        elif model_name in ['svm', 'GBSA']:
            from lifelines import KaplanMeierFitter, NelsonAalenFitter

            # 假设已训练模型 `model`
            risk_scores = model.predict(X_test_permuted)  # 预测风险分数（越大，风险越高）

            # 需要 `y_train`（生存时间和事件），用它来计算基线风险
            times, events = y_train['time'], y_train['status']

            # 计算基线累积风险函数
            naf = NelsonAalenFitter()
            # naf.fit(times, event_observed=events)
            naf.fit(times, event_observed=events, timeline=time_grid) 
            baseline_hazard = naf.cumulative_hazard_

            # 计算生存函数 S(t | X)
            surv_probs_perm = np.exp(-np.outer(np.exp(risk_scores), baseline_hazard.values.flatten()))
        elif model_name in ['super_learner']:
            from lifelines import KaplanMeierFitter, NelsonAalenFitter
            X_meta = get_meta_data(super_learner_fit_models, X_test_permuted, t_median=10.21516891628409) #time from y_train
            # 计算每个样本的风险评分
            risk_scores = model.predict_log_partial_hazard(X_meta).values
            # 需要 `y_train`（生存时间和事件），用它来计算基线风险
            times, events = y_train['time'], y_train['status']

            # 计算基线累积风险函数
            naf = NelsonAalenFitter()
            # naf.fit(times, event_observed=events)
            naf.fit(times, event_observed=events, timeline=time_grid) 
            baseline_hazard = naf.cumulative_hazard_
            # 计算生存函数 S(t | X)
            surv_probs_perm = np.exp(-np.outer(np.exp(risk_scores), baseline_hazard.values.flatten()))

        times, permuted_scores = brier_score(y_train, y_test, surv_probs_perm, time_grid)
        # ratio = permuted_scores / baseline_scores
        feature_importance_results[feature].append(permuted_scores)

        save_data[feature] = permuted_scores

    save_data['full_model'] = baseline_scores
    save_data.to_csv('time_dependent_feature_importance_{}.csv'.format(model_name), index=False)
    # # # 对每个特征，在十折中求平均，得到平均的时变特征重要性曲线
    # avg_feature_importance = {}
    # for feature, diffs in feature_importance_results.items():
    #     avg_diff = np.mean(diffs, axis=0)
    #     avg_feature_importance[feature] = avg_diff

    # 绘制时变特征重要性曲线
    plt.figure(figsize=(10, 6))
    for feature, importance in feature_importance_results.items():
        plt.plot(time_grid, importance[0] * 10, label=feature)
    plt.plot(time_grid, baseline_scores * 10, label="Full Model", linestyle="--", color="black")
    plt.xlabel("Time")
    plt.ylabel("Increase in Brier Score Loss after Permutation")
    plt.title("Time-Dependent Feature Importance Curves")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("time_dependent_feature_importance_curve_{}.png".format(model_name))
    plt.show()



if __name__ == "__main__":
    """
    Data processing
    """
    #svm feature
    select_cols = [
        "Sex",
        "Age",
        "H",
        "W",
        "BMI",
        "Bp",
        "Hr",
        "Alcohol",
        "ESR",
        "Pre-existing condition",
        "smoking",
        "Diabetes_time",
        "CRP",
        "Lesion site",
        "HbA1c",
        "FBG",
        "ALT",
        "AST",
        "ALT_AST",
        "Ua",
        "GFR",
        "TP",
        "ALB",
        "A_G",
        "WBC",
        "D_Dimer",
        "FDP",
        "First_symptoms",
    ]
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
    ) = preprocess_data(data=data, select_cols=select_cols)

    external_data_path = './data/external_data_noise.csv'
    external_data = load_data(external_data_path)
    (external_X_test_scaled, external_X_test, external_y_test, external_scaler) = preprocess_external_data(
        data=external_data, select_cols=select_cols
        )

    # from IPython import embed;embed()
    # exit()
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

    super_learner_fit_models = None
    models, super_learner_fit_models = specific_super_learner_training(
    models=models,
    candidate_model_name=('CoxPH', 'GBSA'),
    X_train=X_train,
    y_train_df=y_train_df,
    X_test=X_test,
    y_test=y_test,
    scaler=scaler,
    y_train=y_train)

    

    # 使用示例

    # plot_rfs_calibration(models['RSF'],'RSF' ,X_test_scaled, y_test, max_time=120)
    # plot_rfs_calibration(models['svm'],'svm' ,X_test_scaled, y_test, max_time=120)
    # plot_rfs_calibration(models['DeepHit'],'DeepHit' ,X_test_scaled, y_test, max_time=120)
    # plot_rfs_calibration(models['NMTLR'],'NMTLR' ,X_test_scaled, y_test, max_time=120)
    # plot_rfs_calibration(models['super_learner'],'super_learner' ,X_test_scaled, y_test, max_time=120, super_learner_fit_models=super_learner_fit_models)

    # plot_calibration(models,X_test_scaled, y_test, max_time=24, super_learner_fit_models=super_learner_fit_models)
    # plot_calibration(models,X_train_scaled, y_train, max_time=24, super_learner_fit_models=super_learner_fit_models)
    plot_calibration(models,external_X_test_scaled, external_y_test , max_time=24, super_learner_fit_models=super_learner_fit_models)

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
        # if name in ['DeepSurv', 'DeepHit', 'svm', 'NMTLR']:
        #     continue
        # if not name in ['svm']:
        #     continue
        print(f"Evaluating {name}...")
        all_results[name] = {}
        for time_point in time_points:
            print(f"  Time point: {time_point}-year")
            if name in ["CoxPH"]:
                res = evaluate_model_at_time(model, name, X_test_scaled, y_test, time_point, super_learner_fit_models=None)
            else:
                # res = evaluate_model_at_time(
                #     model, name, X_test_scaled, y_test, time_point, super_learner_fit_models=super_learner_fit_models
                # )
                # res = evaluate_model_at_time(
                #     model, name, X_train_scaled, y_train, time_point, super_learner_fit_models=super_learner_fit_models
                # )
                res = evaluate_model_at_time(
                    model, name, external_X_test_scaled, external_y_test, time_point, super_learner_fit_models=super_learner_fit_models
                )
            all_results[name][time_point] = res
            print(
                f"    C-index: {res.get('C-index', 'N/A')}, AUC: {res.get('AUC', 'N/A')}"
            )

    
    # 绘制所有模型的 ROC 曲线（按时间点）
    for time_point in time_points:
        plt.figure(figsize=(8, 6))
        
        for name, res in all_results.items():
            save_data =  pd.DataFrame()
            if (
                time_point in res
                and "ROC" in res[time_point]
                and not np.isnan(res[time_point]["ROC"][0]).any()
            ):
                fpr, tpr, roc_auc = res[time_point]["ROC"]
                auc_ci = res[time_point]['AUC_CI']
                # from IPython import embed;embed()
                # exit()
                plt.plot(fpr, tpr, lw=2, label="{} (AUC at {} year: roc_auc {:.3f} (95% CI {:.3f}-{:.3f})".format(name, int(time_point / 12), roc_auc, auc_ci[0], auc_ci[1]))
                
                save_data_key = "{}_{}_{}".format(name, roc_auc, auc_ci)
                print(time_point, name, fpr.shape, tpr.shape)
                save_data[save_data_key + '_fpr'] = fpr
                save_data[save_data_key + '_tpr'] = tpr

                save_data.to_csv('auc_external_val_{}_{}.csv'.format(name, time_point), index=False)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves for All Models at {time_point / 12}-year")
        plt.legend(loc="lower right")
        plt.savefig(f"roc_all_models_{time_point / 12}year.png")
        plt.close()
    
    


    y_train_df = y_train_df.reset_index(drop=True)
    train_data_df = pd.concat((X_train_scaled, y_train_df), axis=1)

    y_test_df = y_test_df.reset_index(drop=True)
    test_data_df = pd.concat((X_test_scaled, y_test_df), axis=1)


    # time_dependent_feature_importance_curve(model=super_learner_fit_models['CoxPH'],
    #                                     model_name='CoxPH',
    #                                         train_data=train_data_df,
    #                                         test_data=test_data_df,
    #                                         select_cols=select_cols)
    
    time_dependent_feature_importance_curve(model=super_learner_fit_models['GBSA'],
                                            model_name='GBSA',
                                                train_data=train_data_df,
                                                test_data=test_data_df,
                                                select_cols=select_cols)

    time_dependent_feature_importance_curve(model=models['super_learner'],
                                    model_name='super_learner',
                                        train_data=train_data_df,
                                        test_data=test_data_df,
                                        select_cols=select_cols)