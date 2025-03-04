import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from config import SEED as seed

def load_data(file_path):
    # columns = [
    #     'Race', 'Marital', 'Year', 'Age', 'Gender', 'Histological type', 
    #     'Primary site', 'Stage', 'Grade', 'Surgery', 'Radiotherapy', 
    #     'Chemotherapy', 'Tumor size', 'Number of tumors', 'Tumor extension',
    #     'Distant metastasis', 'Survival months', 'Status', 'Laterality',
    #     'Regional LN surgery', 'Systemic therapy', 'Metastasis to bone',
    #     'T stage', 'N stage', 'M stage', 'Metastasis to brain/liver/lung', 'income',
    # ]
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data.BMI = data.W * 100 * 100 / data.H / data.H
    data.ALT_AST = data.ALT / data.AST
#     select_cols = covariates = ['Age','ESR', 'Diabetes_time', 'Hb', 'HbA1c', 'FBG','Cr', 'RBC', 'D_Dimer','time', 'Status'
#  ]
  
#     data = data[select_cols]
    X = data.drop(['time', 'Status'], axis=1)
    y = data[['time', 'Status']].rename(columns={'time': 'time', 'Status': 'status'})
    
    X_train, X_test, y_train_df, y_test_df = train_test_split(X, y, test_size=0.3, random_state=seed)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    y_train = Surv.from_arrays(event=y_train_df['status'], time=y_train_df['time'])
    y_test = Surv.from_arrays(event=y_test_df['status'], time=y_test_df['time'])
    
    return X_train_scaled, X_train, y_train, X_test_scaled, X_test, y_test, y_train_df, y_test_df,scaler
