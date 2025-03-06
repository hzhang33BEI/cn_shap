import pandas as pd
import numpy as np

# 设置随机种子保证可复现性
np.random.seed(42)

# ----------------------
# 1. 读取数据
# ----------------------
input_path = "./data/external_data.csv"  # 替换为你的输入文件路径
output_path = "./data/external_data_noise.csv"  # 输出文件路径

# 读取原始数据
df = pd.read_csv(input_path)


# ----------------------
# 2. 定义扰动函数
# ----------------------
def perturb_continuous(column, noise_scale=0.1):
    """为连续变量添加高斯噪声"""
    noise = np.random.normal(loc=0, scale=noise_scale, size=len(column))
    return column + noise


def perturb_categorical(column, perturb_prob=0.1):
    """扰动分类变量（随机替换类别）"""
    categories = column.unique()
    perturbed = column.copy()

    # 生成扰动位置
    mask = np.random.rand(len(column)) < perturb_prob

    for i in np.where(mask)[0]:
        # 排除原值的候选类别
        available_cats = [c for c in categories if c != column.iloc[i]]
        if available_cats:  # 确保有候选类别可用
            perturbed.iloc[i] = np.random.choice(available_cats)

    return perturbed


# ----------------------
# 3. 处理数据
# ----------------------
# 分离不变列
static_cols = df[["time"]]
df_to_perturb = df.drop(columns=[ "time"])

# 自动识别列类型（可根据需要手动修改）
continuous_cols = [
    "Age",
    "H",
    "W",
    "BMI",
    "Hr",
    "ESR",
    "Diabetes_time",
    "CRP",
    "Hb",
    "HbA1c",
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
]
categorical_cols = [
    "Sex",
    "Bp",
    "Alcohol",
    "Pre-existing condition",
    "smoking",
    "Lesion site",
    "First_symptoms",
    
]

# 应用扰动
df_perturbed = pd.concat(
    [
        static_cols,
        df_to_perturb[continuous_cols].apply(perturb_continuous, noise_scale=0.5),
        df_to_perturb[categorical_cols].apply(perturb_categorical, perturb_prob=0.2),
        df_to_perturb[['Status']].apply(perturb_categorical, perturb_prob=0.1),
    ],
    axis=1,
)

# 保持原始列顺序
df_perturbed = df_perturbed[df.columns.tolist()]

# # ----------------------
# # 4. 验证结果
# # ----------------------
# # 检查不变列是否被修改
# try:
#     pd.testing.assert_frame_equal(
#         df_perturbed[["Status", "time"]], df[["Status", "time"]]
#     )
#     print("✅ Status 和 time 列未发生变化")
# except AssertionError:
#     print("❌ 错误：Status 或 time 列被意外修改！")

# ----------------------
# 5. 保存数据
# ----------------------
df_perturbed.to_csv(output_path, index=False)
print(f"📁 扰动后的数据已保存至：{output_path}")

# 显示前3行对比示例
print("\n原始数据前3行：")
print(df.head(3))
print("\n扰动后数据前3行：")
print(df_perturbed.head(3))
