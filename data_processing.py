import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tabulate import tabulate
from lifelines import CoxPHFitter

def get_data():
    # 读取数据
    # origin_data = pd.read_csv('./data/data_raw.csv')
    origin_data = pd.read_csv("data/augmented_12.1.csv")
    origin_data.BMI = origin_data.W * 100 * 100 / origin_data.H / origin_data.H
    origin_data.ALT_AST = origin_data.ALT / origin_data.AST

    return origin_data


def plot_distribution(data):
    fig = plt.figure(figsize=(35, 35))
    gs = GridSpec(7, 5, figure=fig)
    
    for i, colname in enumerate(data.columns):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        
        # 判断数据类型
        if pd.api.types.is_numeric_dtype(data[colname]):  # 连续变量
            sns.histplot(data[colname], bins=20, kde=True, ax=ax)
        else:  # 分类变量
            sns.countplot(x=data[colname], data=data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
            # 计算每个类别的数量
            total = len(data[colname])
            for p in ax.patches:
                # 计算每个类别的百分比
                percentage = '{:.1f}%'.format(100 * p.get_height() / total)
                # 在柱子上方添加百分比文本
                ax.annotate(percentage,  # 要显示的文本
                            (p.get_x() + p.get_width() / 2., p.get_height()),  # 文本的位置
                            ha='center', va='center',  # 水平和垂直对齐方式
                            fontsize=10,  # 字体大小
                            color='black',  # 字体颜色
                            xytext=(0, 5),  # 文本相对于位置的偏移量
                            textcoords='offset points')  # 文本坐标的参考系统

        ax.set_title(colname)
    
    plt.tight_layout()
    plt.savefig('features_distribution.png', dpi=300, bbox_inches="tight")
    # plt.show()


# 获取数据
data = get_data()
# from IPython import embed;embed()
# exit()

# 绘制分布图
plot_distribution(data)


def describe_data(df):
    # 初始化一个空字典来存储结果
    result = {}
    # 遍历数据框的每一列
    for col in df.columns:
        # 检查列的数据类型是否为数值类型
        if pd.api.types.is_numeric_dtype(df[col]):
            # 计算均值和方差
            mean_val = df[col].mean()
            var_val = df[col].var()
            # 格式化均值和方差，保留一位小数
            stat = f"{mean_val:.1f} ({var_val:.1f})"
        else:
            # 计算每个种类的百分比
            percentages = df[col].value_counts(normalize=True)
            # 将百分比格式化为字符串，保留一位小数
            stat = ', '.join([f"{cat}: {perc * 100:.1f}%" for cat, perc in percentages.items()])
        # 将结果存储到字典中
        result[col] = stat
    # 将字典转换为数据框
    desc = pd.DataFrame.from_dict(result, orient='index', columns=['statistic'])
    return desc


overall = describe_data(data)
print("\n总体分布:")
print(tabulate(overall, headers='keys', tablefmt='psql'))

data_cox = data.copy()


# 自定义单变量分析函数
def univariate_cox_analysis(data, duration_col='time', event_col='Status'):
    results = []
    for col in data.columns:
        if col in [duration_col, event_col]: 
            continue
        # print(col)
        # a = data_cox[col].unique()
        # mapping = dict(zip(a , range(len(a))))
        # print(mapping)
        # data_cox[col] = data_cox[col].map(mapping)

        cph = CoxPHFitter()
        try:
            cph.fit(data[[duration_col, event_col, col]], duration_col=duration_col, event_col=event_col)
            hr = cph.summary.loc[col, 'exp(coef)']
            p = cph.summary.loc[col, 'p']
            # print(hr, p)
            # from IPython import embed;embed()
            # exit()
            results.append({
                'Variable': col,
                'HR (95% CI)': f"{hr:.3f} ({cph.confidence_intervals_.loc[col, '95% lower-bound']:.3f}-{cph.confidence_intervals_.loc[col, '95% upper-bound']:.3f})",
                'p-value': f"{p:.3f}"
            })
        except:
            print(col)
            continue
    
    return pd.DataFrame(results)
# from IPython import embed;embed()
# exit(0)
unicox = univariate_cox_analysis(data_cox)
print("\n单变量 Cox 回归:")

print(tabulate(unicox, headers='keys', tablefmt='psql', showindex=False))

# from IPython import embed;embed()
# exit()
#### 多变量 Cox 回归分析 ####
# 准备公式（排除生存时间和状态列）
# covariates = [col for col in data_cox.columns if col not in ['Survival months', 'Status']]
covariates = ['ESR', 'Diabetes_time', 'Hb', 'HbA1c', 'FBG','Cr', 'RBC', 'D_Dimer',
 ]
cph = CoxPHFitter()
# from IPython import embed;embed()
# exit()
cph.fit(data_cox[['time', 'Status'] + covariates], 
       duration_col='time', 
       event_col='Status')

# 生成结果表格
multicox = cph.summary[['exp(coef)', 'p']]
multicox['HR (95% CI)'] = [
    f"{row['exp(coef)']:.3f} ({cph.confidence_intervals_.loc[idx, '95% lower-bound']:.3f}-{cph.confidence_intervals_.loc[idx, '95% upper-bound']:.3f})"
    for idx, row in multicox.iterrows()
]
multicox = multicox[['HR (95% CI)', 'p']].reset_index().rename(columns={'index': 'Variable'})
print("\n多变量 Cox 回归:")
print(tabulate(multicox, headers='keys', tablefmt='psql', showindex=False))

# data_cox.to_csv('./data/data_cox3.csv', index=False)
# data_cox.to_csv('./data/data_cox_income.csv', index=False)
