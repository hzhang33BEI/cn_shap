import pandas as pd
import os
import numpy as np

def get_superlearner_auc_ci_from_str(str):
    auc = float(str.split('_')[2])
    ci = str.split('_')[3].replace('(', '').replace(')','').split(',')
    ci = [float(c) for c in ci]
    return auc, ci[0], ci[1]

def get_auc_ci_from_str(str):
    auc = float(str.split('_')[1])
    ci = str.split('_')[2].replace('(', '').replace(')','').split(',')
    ci = [float(c) for c in ci]
    return auc, ci[0], ci[1]
        
if __name__ == "__main__":
    split = 'val'
    data_root = "model_results/auc/{}/".format(split)
    models = ["DeepHit", "CoxPH", "RSF", "GBSA", "NMTLR","svm","super_learner"]
    months = [12, 24]
    

    for month_ in months:
        
        
        deephit_file_path = os.path.join(data_root, 'auc_{}_{}_{}.csv'.format(split, 'DeepHit', month_))
        deephit = pd.read_csv(deephit_file_path)
        
        nmtlr_file_path = os.path.join(data_root, 'auc_{}_{}_{}.csv'.format(split, 'NMTLR', month_))
        nmtlr = pd.read_csv(nmtlr_file_path)
        
        rsf_file_path = os.path.join(data_root, 'auc_{}_{}_{}.csv'.format(split, 'RSF', month_))
        rsf = pd.read_csv(rsf_file_path)
        
        svm_file_path = os.path.join(data_root, 'auc_{}_{}_{}.csv'.format(split, 'svm', month_))
        svm = pd.read_csv(svm_file_path)

        coxph_file_path = os.path.join(data_root, 'auc_{}_{}_{}.csv'.format(split, 'CoxPH', month_))
        coxph = pd.read_csv(coxph_file_path)
        
        gbsa_file_path = os.path.join(data_root, 'auc_{}_{}_{}.csv'.format(split, 'GBSA', month_))
        gbsa = pd.read_csv(gbsa_file_path)
        
        superlearner_file_path = os.path.join(data_root, 'auc_{}_{}_{}.csv'.format(split, 'super_learner', month_))
        superlearner = pd.read_csv(superlearner_file_path)
        
        nmtlr_str = nmtlr.columns[0]
        nmtlr_auc, nmtlr_ci_lower, nmtlr_ci_upper = get_auc_ci_from_str(nmtlr_str)
        
        deephit_str = deephit.columns[0]
        deephit_auc, deephit_ci_lower, deephit_ci_upper = get_auc_ci_from_str(deephit_str)
        
        rsf_str = rsf.columns[0]
        rsf_auc, rsf_ci_lower, rsf_ci_upper = get_auc_ci_from_str(rsf_str)
        
        svm_str = svm.columns[0]
        svm_auc, svm_ci_lower, svm_ci_upper = get_auc_ci_from_str(svm_str)
        

        coxph_str = coxph.columns[0]
        coxph_auc, coxph_ci_lower, coxph_ci_upper = get_auc_ci_from_str(coxph_str)
        
        gbsa_str = gbsa.columns[0]
        gbsa_auc, gbsa_ci_lower, gbsa_ci_upper = get_auc_ci_from_str(gbsa_str)
        
        
        # from IPython import embed;embed()
        # exit()
        superlearner_str = superlearner.columns[0]
        superlearner_auc, superlearner_ci_lower, superlearner_ci_upper = get_superlearner_auc_ci_from_str(superlearner_str)
        
        
        deephit_length = deephit.shape[0]
        nmtlr_length = nmtlr.shape[0]
        rsf_length = rsf.shape[0]
        svm_length = svm.shape[0]
        superlearner_length = superlearner.shape[0]
        coxhp_length = coxph.shape[0]
        gbsa_length = gbsa.shape[0]
        
        # all_result = pd.DataFrame()
        # all_result['1-Specifity'] = pd.concat([deephit.iloc[:,0], nmtlr.iloc[:,0], rsf.iloc[:,0], svm.iloc[:,0], superlearner.iloc[:,0]], axis=0, ignore_index=True)
        


        # 定义列名和行数
        columns = ['1-Specifity',
                   "DeepHit:{:.3f}(95%CI:{:.3f}-{:.3f})".format(deephit_auc, deephit_ci_lower, deephit_ci_upper),
                   "NMTLR:{:.3f}(95%CI:{:.3f}-{:.3f})".format(nmtlr_auc, nmtlr_ci_lower, nmtlr_ci_upper),
                   "RSF:{:.3f}(95%CI:{:.3f}-{:.3f})".format(rsf_auc, rsf_ci_lower, rsf_ci_upper),
                   "FSSVM:{:.3f}(95%CI:{:.3f}-{:.3f})".format(svm_auc, svm_ci_lower, svm_ci_upper),
                   "CoxPH:{:.3f}(95%CI:{:.3f}-{:.3f})".format(coxph_auc, coxph_ci_lower, coxph_ci_upper),
                   "GBSA:{:.3f}(95%CI:{:.3f}-{:.3f})".format(gbsa_auc, gbsa_ci_lower, gbsa_ci_upper),
                   "Super Learner:{:.3f}(95%CI:{:.3f}-{:.3f})".format(superlearner_auc, superlearner_ci_lower, superlearner_ci_upper),
                   ]
        rows = deephit_length + nmtlr_length + rsf_length + svm_length + superlearner_length + coxhp_length + gbsa_length

        # from IPython import embed;embed()
        # exit()
        # 创建空 DataFrame，填充 NaN
        empty_df = pd.DataFrame(np.nan, index=range(rows), columns=columns)
        empty_df['1-Specifity'] = pd.concat([deephit.iloc[:,0], nmtlr.iloc[:,0], rsf.iloc[:,0], svm.iloc[:,0],coxph.iloc[:,0],gbsa.iloc[:,0],superlearner.iloc[:,0]], axis=0, ignore_index=True)
        empty_df.iloc[:deephit_length,1] = deephit.iloc[:,1]
        empty_df.iloc[deephit_length:deephit_length+nmtlr_length,2] = nmtlr.iloc[:,1]
        empty_df.iloc[deephit_length+nmtlr_length:deephit_length+nmtlr_length+rsf_length,3] = rsf.iloc[:,1]
        empty_df.iloc[deephit_length+nmtlr_length+rsf_length:deephit_length+nmtlr_length+rsf_length+svm_length,4] = svm.iloc[:,1]
        empty_df.iloc[deephit_length+nmtlr_length+rsf_length+svm_length:deephit_length+nmtlr_length+rsf_length+svm_length+coxhp_length,5] = coxph.iloc[:,1]
        empty_df.iloc[deephit_length+nmtlr_length+rsf_length+svm_length+coxhp_length:deephit_length+nmtlr_length+rsf_length+svm_length+coxhp_length+gbsa_length,6] = gbsa.iloc[:,1]
        empty_df.iloc[deephit_length+nmtlr_length+rsf_length+svm_length+coxhp_length+gbsa_length:deephit_length+nmtlr_length+rsf_length+svm_length+coxhp_length+gbsa_length+superlearner_length,7] = superlearner.iloc[:,1] 
        print(empty_df)
        
        empty_df.to_csv("auc_for_prism_{}_{}.csv".format(split, month_), index=False)