from datasets import load_data, preprocess_data, preprocess_external_data
from models import train_base_models
from model_evaluation import evaluate_model, evaluate_model_at_time
from matplotlib import pyplot as plt
import numpy as np
from super_learner import models_super_learner
from sklearn.preprocessing import StandardScaler

import logging
from datetime import datetime

selected_features = {
    "CoxPH": [
        "Sex",
        "Age",
        "H",
        "W",
        "BMI",
        "Bp",
        "Hr",
        "Alcohol",
        "Pre-existing condition",
        "smoking",
        "Diabetes_time",
        "Lesion site",
        "Hb",
        "FBG",
        "ALT",
        "AST",
        "ALT_AST",
        "Ua",
        "Cr",
        "GFR",
        "TP",
        "ALB",
        "A_G",
        "RBC",
        "WBC",
        "PLT",
        "FDP",
        "First_symptoms",
    ],
    "DeepSurv": [
        "Sex",
        "Age",
        "H",
        "W",
        "BMI",
        "Bp",
        "Hr",
        "Alcohol",
        "Pre-existing condition",
        "smoking",
        "Diabetes_time",
        "Lesion site",
        "Hb",
        "HbA1c",
        "ALT",
        "AST",
        "ALT_AST",
        "Ua",
        "Cr",
        "GFR",
        "TP",
        "ALB",
        "A_G",
        "RBC",
        "WBC",
        "PLT",
        "D_Dimer",
        "FDP",
        "First_symptoms",
    ],
    "DeepHit": [
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
        "Hb",
        "FBG",
        "ALT",
        "AST",
        "ALT_AST",
        "Cr",
        "GFR",
        "ALB",
        "A_G",
        "RBC",
        "PLT",
        "D_Dimer",
        "FDP",
        "First_symptoms",
    ],
    "NMTLR": [
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
        "Ua",
        "Cr",
        "GFR",
        "TP",
        "ALB",
        "A_G",
        "RBC",
        "D_Dimer",
        "FDP",
        "First_symptoms",
    ],
    "svm": [
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
    ],
    "RSF": [
        "Sex",
        "Age",
        "H",
        "W",
        "BMI",
        "Bp",
        "Hr",
        "Alcohol",
        "Pre-existing condition",
        "smoking",
        "Diabetes_time",
        "CRP",
        "Lesion site",
        "Hb",
        "FBG",
        "ALT",
        "AST",
        "Ua",
        "Cr",
        "GFR",
        "TP",
        "ALB",
        "A_G",
        "RBC",
        "WBC",
        "PLT",
        "D_Dimer",
        "FDP",
        "First_symptoms",
    ],
    "GBSA": [
        "Sex",
        "Age",
        "H",
        "W",
        "BMI",
        "Bp",
        "Hr",
        "Alcohol",
        "Pre-existing condition",
        "smoking",
        "Diabetes_time",
        "Lesion site",
        "Hb",
        "FBG",
        "ALT",
        "AST",
        "ALT_AST",
        "Ua",
        "Cr",
        "GFR",
        "TP",
        "ALB",
        "A_G",
        "RBC",
        "WBC",
        "PLT",
        "D_Dimer",
        "FDP",
        "First_symptoms",
    ],
}


# 配置日志
log_filename = f"logs/log_model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    """
    Data processing
    """
    # data_path = 'data/data_cox1.csv'
    data_path = "data/augmented_12.1.csv"
    externam_data_path = "./data/external_data_noise.csv"

    for feature_name, selected_feature_ in selected_features.items():
        logging.info(f"Processing feature: {feature_name} {selected_feature_}")
        logging.info("{}".format("*" * 100))
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
        ) = preprocess_data(data=data, select_cols=selected_feature_)

        external_data = load_data(externam_data_path)
        (external_X_test_scaled, external_X_test, external_y_test, external_scaler) = preprocess_external_data(
            data=external_data, select_cols=selected_feature_
        )

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

        # 评估所有模型
        # results = {}
        for name, model in models.items():
            # logging.info(f"Evaluating {name}...")
            if name in ["CoxPH", "AFT"]:
                res_val = evaluate_model(model, name, X_test, y_test)
                res_train = evaluate_model(model, name, X_train, y_train)
                external_res_val = evaluate_model(model, name, external_X_test, external_y_test)
            else:
                res_val = evaluate_model(model, name, X_test_scaled, y_test)
                res_train = evaluate_model(model, name, X_train_scaled, y_train)
                external_res_val = evaluate_model(model, name, external_X_test_scaled, external_y_test)
            # results[name] = res
            # # 修改3：添加格式化字符串中的指标名称更清晰
            # log_msg = f"{name} Evaluation - " f"Val-C-index: {res_val['C-index']:.4f}, "
            log_msg = "{} -> Train-C-index: {} Internal-Val-C-index: {} External-Val-C-index: {}".format(
                name, res_train["C-index"], res_val["C-index"], external_res_val["C-index"]
            )
            logging.info(log_msg)

        # Super learner
        models_super_learner(
            models=models,
            X_train=X_train,
            y_train_df=y_train_df,
            X_test=X_test,
            y_test=y_test,
            scaler=scaler,
            y_train=y_train,
            external_X_test=external_X_test,
            external_y_test=external_y_test,
            external_scaler=external_scaler
        )
