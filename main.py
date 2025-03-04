from datasets import load_data, preprocess_data
from models import train_base_models
from model_evaluation import evaluate_model, evaluate_model_at_time
from matplotlib import pyplot as plt
import numpy as np
from super_learner import models_super_learner
from sklearn.preprocessing import StandardScaler


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
    # # 定义评估的时间点
    # time_points = [1, 3, 5, 10]   # 1-year, 3-year, 5-year, 10-year
    # time_points = [t * 12 for t in time_points]

    # # 评估所有模型并收集结果
    # all_results = {}
    # for name, model in models.items():
    #     print(f"Evaluating {name}...")
    #     all_results[name] = {}
    #     for time_point in time_points:
    #         print(f"  Time point: {time_point}-year")
    #         if name in ['CoxPH']:
    #             res = evaluate_model_at_time(model, name, X_test, y_test, time_point)
    #         else:
    #             res = evaluate_model_at_time(model, name, X_test_scaled, y_test, time_point)
    #         all_results[name][time_point] = res
    #         print(f"    C-index: {res.get('C-index', 'N/A')}, AUC: {res.get('AUC', 'N/A')}")


    # # 绘制所有模型的 ROC 曲线（按时间点）
    # for time_point in time_points:
    #     plt.figure(figsize=(8, 6))
    #     for name, res in all_results.items():
    #         if time_point in res and 'ROC' in res[time_point] and not np.isnan(res[time_point]['ROC'][0]).any():
    #             fpr, tpr, roc_auc = res[time_point]['ROC']
    #             # from IPython import embed;embed()
    #             # exit()
    #             plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'ROC Curves for All Models at {time_point / 12}-year')
    #     plt.legend(loc="lower right")
    #     plt.savefig(f'roc_all_models_{time_point / 12}year.png')
    #     plt.close()


    # # 绘制所有模型的校准曲线（按时间点）
    # for time_point in time_points:
    #     plt.figure(figsize=(8, 6))
    #     for name, res in all_results.items():
    #         if time_point in res and 'Calibration' in res[time_point] and not np.isnan(res[time_point]['Calibration'][0]).any():
    #             prob_true, prob_pred = res[time_point]['Calibration']
    #             plt.plot(prob_pred, prob_true, marker='o', label=name)
    #     plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    #     plt.xlabel('Predicted Probability')
    #     plt.ylabel('True Probability')
    #     plt.title(f'Calibration Curves for All Models at {time_point / 12}-year')
    #     plt.legend()
    #     plt.savefig(f'calibration_all_models_{time_point / 12}year.png')
    #     plt.close()


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
    # print("\nFinal Evaluation Results:")
    # for name, res in results.items():
    #     print(f"{name}:")
    #     print(f"  C-index: {res['C-index']:.4f}")
    #     print(f"  AUC: {res.get('AUC', 'N/A'):.4f}")
    #     if 'Calibration' in res:
    #         print(f"  Calibration: {res['Calibration'][0].mean():.4f}")


