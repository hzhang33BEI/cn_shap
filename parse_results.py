import re
import pandas as pd


if __name__ == "__main__":
    data_path = './logs/log_model_training_20250306_111052.log'
    
    current_feature = None
    
    result_save_path = './data/results.csv'
    df = pd.DataFrame(columns=["feature", "model", "Train-C-index", "Internal-Val-C-index", "External-Val-C-index", 'Mean-Val-C-index'])

    
    with open(data_path, 'r') as reader:
        for line in reader:
            line = line.strip()
            
            if 'Processing feature:' in  line:
                current_feature = line.split('Processing feature: ')[1].split(' [')[0]
            
            
            if ' -> ' in line:
                if 'Super Learner ' in line:
                    models = re.search(r"Super Learner -> \((.*?)\)", line).group(1)
                    model_name = [m.strip("' ") for m in models.split(',')]


                else:
                    model_match = re.search(r"-\s([\w\s-]+)\s->", line)
                    model_name = model_match.group(1).strip() if model_match else None
                    model_name = model_name.replace('INFO - ', '')

                # 提取C-index值
                # cindex_matches = re.findall(r"(\w+-C-index): (\d+\.\d+)", line)
                # cindex_dict = dict(cindex_matches)
                # from IPython import embed;embed()
                # exit()
                
                Internal_Val_C_index= float(re.findall(r"(Internal-Val-C-index): (\d+\.\d+)", line)[0][1])
                External_Val_C_index= float(re.findall(r"(External-Val-C-index): (\d+\.\d+)", line)[0][1])
                Train_C_index = float(re.findall(r"(Train-C-index): (\d+\.\d+)", line)[0][1])
                # 转换为浮点数
                result = {
                    "feature": current_feature,
                    "model": model_name,
                    "Train-C-index":  Train_C_index, 
                    "Internal-Val-C-index": Internal_Val_C_index,
                    "External-Val-C-index": External_Val_C_index,
                    "Mean-Val-C-index": (Internal_Val_C_index + External_Val_C_index) / 2.0
                    }
                df = df.append(result, ignore_index=True)
    df.to_csv(result_save_path, index=False)
                
                