# ------------------------- config.py -------------------------
import os
import numpy as np

# 全局配置
SEED = 42
DATA_PATH = 'data/data_cox1.csv'
TIME_POINTS = [12, 36, 60, 120]  # 1,3,5,10年（单位：月）
best_params_deephit = {'lr': 0.0001, 'num_epochs': 100, 'batch_size': 8, 'num_nodes': [64, 64], 'dropout': 0.2, 'num_durations': 36} 
best_params_nmtlr = {'lr': 0.001, 'num_epochs': 100, 'batch_size': 8, 'num_nodes': [32, 32], 'dropout': 0.2, 'num_durations': 36}
best_params_ds = {'lr': 0.001, 'num_epochs': 100, 'optimizer': 'adam'} 
best_params_xgb = {
        'objective': 'survival:cox', 
        'eval_metric': 'cox-nloglik', 
        'max_depth': 6,  # 你可以根据需要调整超参数
        'eta': 0.1,
        'colsample_bytree': 0.8,
        'subsample': 0.7
    }
# 数据列配置
SELECT_COLS = [
    'Race', 'Marital', 'Age', 'Gender', 'Histological type',
    'Primary site', 'Grade', 'Surgery', 'Radiotherapy',
    'Chemotherapy', 'Tumor size', 'Number of tumors',
    'Distant metastasis', 'Survival months', 'Status', 'M stage'
]

# 超参数网格
PARAM_GRIDS = {
    'DeepSurv': {
        'lr': [1e-3, 1e-4],
        'num_epochs': [200, 400],
        'optimizer': ['adam']
    },
    'DeepHit': {
        'lr': [1e-3, 1e-4],
        'num_epochs': [200],
        'batch_size': [256, 512],
        'num_nodes': [[64, 64], [32, 32]],
        'dropout': [0.1, 0.2],
        'num_durations': [12, 24]
    },
    'NMTLR': {
        'lr': [1e-3, 1e-4],
        'num_epochs': [200],
        'batch_size': [256, 512],
        'num_nodes': [[32, 32], [64, 64]],
        'dropout': [0.1, 0.2],
        'num_durations': [12, 24]
    }
}

# 设置全局随机种子
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)