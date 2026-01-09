# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 15:08:22 2026

@author: CHENGXIMEI
"""



import anndata as ad
import networkx as nx
import scanpy as sc
# import scglue
from matplotlib import rcParams
import os
import scanpy as sc
import pandas as pd
import numpy as np
import os
import random 
random.seed(123)
data_dir = "/media/dell/NETAC/sq2_scp_ramdom_forest1224"
os.chdir(data_dir)

# %% read data

rna_sq2=ad.read_h5ad("sub_sce_aftercircos_tosq2.h5ad")
rna_sq2.layers['counts'].max()

rna_scp=ad.read_h5ad("rna_scp_EMB_fix.h5ad")
# rna_scp.obs_names
# rna_scp.obs["Raw.file"]
# rna_scp.obs
# rna_scp.obs_names = (
#     rna_scp.obs_names
#     .str.replace(r"\.raw\.PG\.Quantity$", "", regex=True)
# )

# rna_scp.write("rna_scp_EMB_fix.h5ad")


# %%scp commond cacium build

meta_data = rna_scp.obs.copy()
A_value=pd.to_numeric(meta_data['A_20-2.8_A'], errors='coerce')

S_value=pd.to_numeric(meta_data['S_20-2.8_A'], errors='coerce')
meta_data["S_value"] = pd.to_numeric(meta_data["S_20-2.8_A"], errors="coerce")
meta_data["A_value"] = pd.to_numeric(meta_data["A_20-2.8_A"], errors="coerce")

# %%% 分responcell

A_value.head()
A_value.dtype  # 应该是 float64

meta_data["A_Calcium.Response"] = np.where(A_value > 1,
                                           "Responsive",
                                           "Non-Responsive")

meta_data["S_Calcium.Response"] = np.where(S_value > 1,
                                           "Responsive",
                                           "Non-Responsive")

# %%%scp按照阈值进行划分
meta_data["common_Calcium.Response"] = np.nan

# 3) case_when 等价写法：np.select
# 有一些问题，就是在R中发现最大值的数（A_value）最大值并没有1.2,但是居然还能筛选成功还是很amazing，在这里放低筛选条件

sum(meta_data["S_value"]>1.05)#1.145 21
sum(meta_data["S_value"]<0.93)# 0.84 21



conditions = [
    (S_value >=1.05) ,
    (S_value <0.93)
]

choices = [
    "Responsive",
    "Non-Responsive"
]

meta_data["common_Calcium.Response"] = np.select(
    conditions, choices, default="Weak"
)

print(meta_data["common_Calcium.Response"].value_counts())
print("sq2,ca count:",rna_sq2.obs["common_Calcium.Response"].value_counts())

# sub.sce@meta.data <- meta_data

rna_scp.obs=meta_data.copy()

rna_scp.obs["common_Calcium.Response"]



# %% get common genes
rna_sq2.var_names
rna_scp.var_names


common_genes = set(rna_sq2.var_names) & set(rna_scp.var_names)
common_count = len(common_genes)
common_genes_list = sorted(list(common_genes))

print(f"rna_sq2 总基因数: {rna_sq2.n_vars}")
print(f"rna_scp 总基因数: {rna_scp.n_vars}")
print(f"共同基因数量: {common_count}")

rna_sq2_common = rna_sq2[:, common_genes_list].copy()
rna_scp_common = rna_scp[:, common_genes_list].copy()

# rna_scp_common.obs["S_Pred"]
# rna_sq2_common.obs["S_Calcium.Response"]

# %% process 
import scipy.sparse as sp
import numpy as np
import pandas as pd

common_genes = np.intersect1d(rna_sq2.var_names, rna_scp.var_names)
#%% 提取X
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

# =========================
# 1. 取共同基因
# =========================
genes_sq2 = rna_sq2_common.var_names
genes_scp = rna_scp_common.var_names

common_genes = np.intersect1d(genes_sq2, genes_scp)
print("Common genes:", len(common_genes))


# =========================
# 2. 构建 X（不加前缀）
# =========================
def get_X(adata, genes):
    X = adata[:, genes].layers["scVI_denoised"]
    if sp.issparse(X):
        X = X.toarray()
    return pd.DataFrame(X, index=adata.obs_names, columns=genes)


X_sq2 = get_X(rna_sq2_common, common_genes)
X_scp = get_X(rna_scp_common, common_genes)

# =========================
# 3. 合并 SQ2 + SCP
# =========================
X = pd.concat([X_sq2, X_scp], axis=0)


# =========================
# 4. 添加平台显式变量
# =========================
platform = pd.Series(
    ["SQ2"] * X_sq2.shape[0] + ["SCP"] * X_scp.shape[0],
    index=X.index,
    name="platform"
)

X["platform"] = platform


# =========================
# 5. One-hot 编码平台
# =========================
X = pd.get_dummies(X, columns=["platform"], drop_first=True)
# 得到 platform_SCP


# =========================
# 6. 缺失值处理（保险）
# =========================
X = X.fillna(0)







#%%% 提取Y
from neuroCombat import neuroCombat

y_sq2 = rna_sq2_common.obs["common_Calcium.Response"]
y_scp = rna_scp_common.obs["common_Calcium.Response"]

y = pd.concat([y_sq2, y_scp], axis=0)




valid_idx = y.notna()
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# 去批次
batch = (
    X["platform_SQ2"]
    .map({True: "SQ2", False: "SCP"})
    .astype("category")
)
gene_cols = gene_cols = [
    g for g in common_genes
    if g in X.columns
    and g != "platform_SQ2"
]
assert "platform_SQ2" not in gene_cols
X_gene = X[gene_cols].astype(float)

# 去掉零方差基因（ComBat 必须）
var = X_gene.var(axis=0)
gene_cols = var[var > 0].index.tolist()

X_gene = X_gene[gene_cols]

combat_data = neuroCombat(
    dat=X_gene[gene_cols].T,  # shape: genes × cells
    covars=pd.DataFrame({"batch": batch}),
    batch_col="batch"
)
X_combat = X.copy()
X_combat[gene_cols] = combat_data["data"].T

# =========================
# 7. 标准化
# =========================
scaler = StandardScaler()

platform_col = ["platform_SQ2"]
scaler = StandardScaler()
X_scaled = X_combat.copy()

X_scaled[gene_cols] = scaler.fit_transform(X_combat[gene_cols])

# 去异常值
Z_TH = 5
X_scaled[gene_cols] = X_scaled[gene_cols].clip(
    lower=-Z_TH,
    upper=Z_TH
)

# X_scaled.max()

#%%bayes
from sklearn.model_selection import train_test_split, StratifiedKFold
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Integer, Categorical, Real
import numpy as np

# =========================
# 1. 准备 X / y
# =========================
X_model = X_scaled.drop(columns=["platform_SQ2"])
y_valid = y.loc[X_model.index]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_valid,
    test_size=0.2,
    stratify=y_valid,
    random_state=42
)

# 将平台列转成分组信息
groups_train = X_scaled.loc[X_train.index, "platform_SQ2"].astype(str)  # 'True'/'False'

print("X_train:", X_train.shape, "y_train:", y_train.shape, "groups_train:", groups_train.shape)

# =========================
# 2. 参数空间
# =========================
param_space = {
    "n_estimators": Integer(10, 60),
    'max_depth': Integer(40, 100),
    'min_samples_split': Integer(8, 20),
    'min_samples_leaf': Integer(10, 30),
    'max_features': Categorical(['sqrt', 'log2', 0.3, 0.5, 0.7]),
    'max_samples': Real(0.6, 0.9)
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

# =========================
# 3. CV 改写（小样本安全）
# =========================
# 小样本用 StratifiedKFold，不再强制 group fold
# 保留 stratify y 的信息，n_splits=3 避免 fold 里空样本
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# =========================
# 4. BayesSearchCV
# =========================
opt = BayesSearchCV(
    rf,
    param_space,
    n_iter=100,  # 小样本降低迭代次数更安全
    scoring="roc_auc_ovr",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("开始贝叶斯优化搜索...")
opt.fit(X_train, y_train)  # 不传 groups，避免空样本错误

best_rf = opt.best_estimator_
print("\n" + "="*50)
print("贝叶斯优化搜索完成!")
print("="*50)
print("最佳参数:", opt.best_params_)
print("最佳交叉验证 ROC AUC:", opt.best_score_)

# =========================
# 5. 测试集评估
# =========================
from sklearn.metrics import classification_report, roc_auc_score

y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)

print("\n测试集分类报告:")
print(classification_report(y_test, y_pred))

try:
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    print(f"测试集 ROC AUC (ovr): {auc:.4f}")
except:
    print("ROC AUC 计算失败，可能是样本类别太少。")

#%% save
import joblib
data_dir = "/media/dell/NETAC/sq2_scp_ramdom_forest1224/data/bayes 1-8 200 iter batch"
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)
# 保存模型
joblib.dump(best_rf, "best_random_forest.pkl")
# 如果以后加载
# best_rf_test = joblib.load("best_random_forest.pkl")
joblib.dump(opt, "modol_train.pkl")
import pandas as pd

# 所有迭代结果
opt_results = pd.DataFrame(opt.cv_results_)

# 保存到 csv
opt_results.to_csv("bayes_search_results.csv", index=False)

# 最优参数
best_params = pd.Series(opt.best_params_)
best_params.to_csv("best_rf_params.csv", header=["value"])

type(best_rf)
#%% best-rf to test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# 获取最优参数
best_params = opt.best_params_

# 用训练集训练 RF（可以直接用 best_rf 也可以重新初始化）
rf_test = RandomForestClassifier(
    random_state=42,
    # n_estimators=200,
    n_jobs=-1,
    class_weight="balanced",
    **best_params  # 解包最优参数
)

# 在训练集上拟合
rf_test.fit(X_train, y_train)

# 预测类别
y_pred = rf_test.predict(X_test)

# 预测概率（用于 ROC / AUC）
y_score = rf_test.predict_proba(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#%% MDI
mdi = pd.Series(
    best_rf.feature_importances_,
    index=X_train.columns
)

mdi_genes = mdi.loc[common_genes]


# 检查有多少特征被随机森林使用过

#%% mda

from sklearn.inspection import permutation_importance

perm_global = permutation_importance(
    best_rf,
    X_train,
    y_train,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

mda_global = pd.Series(
    perm_global.importances_mean,
    index=X_train.columns
).loc[common_genes]


joblib.dump(perm_global, "perm_global_model.joblib")

# %%% load

perm_gobal=joblib.load("perm_global_model.joblib")

#%%SQ2 / SCP 平台特异 MDA（关键）
# SQ2
idx_sq2 = X_scaled["platform_SQ2"] == 1
X_sq2_model = X_model.loc[idx_sq2]
y_sq2 = y.loc[idx_sq2]


perm_sq2 = permutation_importance(
    best_rf,
    X_sq2_model,
    y_sq2,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

mda_sq2 = pd.Series(
    perm_sq2.importances_mean,
    index=X_sq2_model.columns
).loc[common_genes]


# SCP
idx_scp = X_scaled["platform_SQ2"] == 0

X_scp_model = X_model.loc[idx_scp]
y_scp = y.loc[idx_scp]
perm_scp = permutation_importance(
    best_rf,
    X_scp_model,
    y_scp,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

mda_scp = pd.Series(
    perm_scp.importances_mean,
    index=X_sq2_model.columns
).loc[common_genes]

top20_mdi = mdi_genes.sort_values(ascending=False).head(20)
top20_sq2 = mda_sq2.sort_values(ascending=False).head(20)
top20_scp = mda_scp.sort_values(ascending=False).head(20)

result = pd.DataFrame({
    "MDI_Gini": mdi_genes,
    "MDA_global": mda_global,
    "MDA_SQ2": mda_sq2,
    "MDA_SCP": mda_scp
})
result.to_csv("feature_importance_results.csv", index=True, header=True)
result_sorted_sq2 = result.sort_values("MDA_SQ2", ascending=False).head(20)
result_sorted_scp = result.sort_values("MDA_SCP", ascending=False).head(20)


joblib.dump(perm_scp, "perm_scp_model.joblib")
joblib.dump(perm_sq2, "perm_sq2_model.joblib")

# %%% load(choose)
perm_sq2=joblib.load("perm_sq2_model.joblib")
perm_scp=joblib.load("perm_scp_model.joblib")

#%% bubble plot
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 设置绘图参数
size = 12

data_dir = "/media/dell/NETAC/sq2_scp_ramdom_forest1224/data/bayes 1-8 200 iter batch/result"
os.makedirs(data_dir, exist_ok=True)
os.chdir(data_dir)

# 假设 top20_sq2 和 top20_scp 是 pd.Series，index 是基因名，values 是 MDA 值
df_sq2 = top20_sq2.sort_values(ascending=True)
df_scp = top20_scp.sort_values(ascending=True)

# 创建 subplot
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=False)

# ======================
# 1. SQ2 bubble plot
# ======================
y_pos_sq2 = range(len(df_sq2))
axes[0].scatter(
    df_sq2.values,
    y_pos_sq2,
    s=df_sq2.values * 500 * size,  # 气泡大小
    color="#1f77b4",  # 蓝色
    alpha=0.7,
    edgecolors="black"
)

axes[0].set_yticks(y_pos_sq2)
axes[0].set_yticklabels(df_sq2.index)
axes[0].set_title("Top 20 SQ2 Genes (Permutation Importance / MDA)")
axes[0].set_xlabel("MDA (Permutation Importance, log scale)")
axes[0].set_xscale('log')  # 对数坐标

# ======================
# 2. SCP bubble plot
# ======================
y_pos_scp = range(len(df_scp))
axes[1].scatter(
    df_scp.values,
    y_pos_scp,
    s=df_scp.values * 500 * size,  # 气泡大小
    color="#ff7f0e",  # 橙色
    alpha=0.7,
    edgecolors="black"
)

axes[1].set_yticks(y_pos_scp)
axes[1].set_yticklabels(df_scp.index)
axes[1].set_title("Top 20 SCP Genes (Permutation Importance / MDA)")
axes[1].set_xlabel("MDA (Permutation Importance, log scale)")
axes[1].set_xscale('log')  # 对数坐标

# 图例（可选）
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='SQ2',
           markerfacecolor='#1f77b4', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='SCP',
           markerfacecolor='#ff7f0e', markersize=10)
]
axes[0].legend(handles=legend_elements, title="Platform", loc="upper right")

plt.tight_layout()

# 保存图
plt.savefig("Top20_SQ2_SCP_Gene_Importances.pdf", format="pdf", bbox_inches="tight")
plt.savefig("Top20_SQ2_SCP_Gene_Importances.svg", format="svg", bbox_inches="tight")

plt.show()

#%%% gobal bubble
import matplotlib.pyplot as plt

# 设置气泡图参数
size_factor = 2000  # 控制气泡大小，可根据数值范围调整
color = "#2ca02c"   # 绿色，可以自定义颜色

# 按重要性升序排列，画图时最重要的在上面
top20_mdi_plot = top20_mdi.sort_values(ascending=True)
y_pos = range(len(top20_mdi_plot))

plt.figure(figsize=(8, 10))

plt.scatter(
    top20_mdi_plot.values,
    y_pos,
    s=top20_mdi_plot.values * size_factor,
    color=color,
    alpha=0.7,
    edgecolors="black"
)

plt.yticks(y_pos, top20_mdi_plot.index)
plt.xlabel("MDI (Gini Importance)")
plt.title("Top 20 Genes by MDI (Random Forest)")

plt.tight_layout()

# 保存图
plt.savefig("Top20_MDI_Genes_Bubble.pdf", format="pdf", bbox_inches="tight")
plt.savefig("Top20_MDI_Genes_Bubble.svg", format="svg", bbox_inches="tight")

plt.show()
#%%ROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import numpy as np

# 获取类别
classes = y_train.unique()  # 或 list(y_train.unique())
#分类下标签二值化
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

# 模型预测概率
y_score = best_rf.predict_proba(X_test)


plt.figure(figsize=(6, 6))

# 为每个类别绘制 ROC
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.3f})')

# 对角线




plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multi-class Random Forest')
plt.legend(loc="lower right")
plt.grid(False)
plt.tight_layout()
plt.savefig(
    "ROC_sq2_scp_bayes.pdf",
    format="pdf",
    bbox_inches="tight"
    )

plt.savefig(
    "ROC_sq2_scp_bayes.svg",
    format="svg",
    bbox_inches="tight"
    )
plt.show()

#%% 上下调
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 假设已有数据：
# - y: 钙响应标签（分类变量）
# - X_scaled: 标准化后的基因表达矩阵
# - common_genes: 共同基因列表
# - mda_sq2, mda_scp: 平台特异的MDA重要性

# 提取标签信息
y_labels = y  # 分类标签

# 按平台分离数据
X_sq2_data = X_scaled[X_scaled["platform_SQ2"] == 1][common_genes]
X_scp_data = X_scaled[X_scaled["platform_SQ2"] == 0][common_genes]

y_sq2 = y_labels[X_scaled["platform_SQ2"] == 1]
y_scp = y_labels[X_scaled["platform_SQ2"] == 0]

top20_sq2_genes = mda_sq2.sort_values(ascending=False).head(20).index.tolist()
top20_scp_genes = mda_scp.sort_values(ascending=False).head(20).index.tolist()

print(f"SQ2 top20 genes: {len(top20_sq2_genes)}")
print(f"SCP top20 genes: {len(top20_scp_genes)}")

def analyze_gene_label_relation_improved(X_data, y_labels, genes_list, platform_name):
    """
    改进版本：能正确识别上下调基因
    """
    results = []
    
    for gene in genes_list:
        gene_data = X_data[gene]
        categories = y_labels.unique()
        
        # 计算每个类别的平均表达
        category_means = {}
        for cat in categories:
            cat_mask = y_labels == cat
            category_means[cat] = gene_data[cat_mask].mean()
        
        # 找出最高和最低表达的类别
        max_cat = max(category_means, key=category_means.get)
        min_cat = min(category_means, key=category_means.get)
        
        # === 关键修改：计算所有类别相对于基线的变化 ===
        # 方法1：计算相对于所有类别平均值的fold change
        overall_mean = gene_data.mean()
        fold_changes = {cat: (category_means[cat] - overall_mean) / overall_mean 
                       for cat in categories}
        
        # 方法2：或者计算相对于某个参考类别（如Non-Responsive）
        reference_cat = 'Non-Responsive'  # 根据您的标签名称调整
        if reference_cat in categories:
            reference_expression = category_means[reference_cat]
            fold_changes_vs_ref = {cat: (category_means[cat] - reference_expression) 
                                  for cat in categories}
        
        # 方法3：或者简单返回每个类别的平均表达
        category_expression = category_means
        
        # ANOVA检验
        groups = [gene_data[y_labels == cat] for cat in categories]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # 判断表达模式
        # 例如：如果最高表达在"Non-Responsive"，最低在"Responsive"
        # 则意味着在Responsive中下调
        
        results.append({
            'Gene': gene,
            'Max_Category': max_cat,
            'Min_Category': min_cat,
            'Max_Expression': category_means[max_cat],
            'Min_Expression': category_means[min_cat],
            # 'Fold_Change': category_means[max_cat] - category_means[min_cat],
            'Fold_Change': fold_changes,
            'Category_Means': category_means,  # 保存所有类别表达
            'Category_Means': category_means,  # 保存所有类别表达
            'Is_Downregulated': max_cat in ['Non-Responsive', 'Weak'],  # 示例条件
            'ANOVA_p': p_value,
            'ANOVA_F': f_stat
        })
    
    return pd.DataFrame(results)
# 执行分析
sq2_relation = analyze_gene_label_relation_improved(X_sq2_data, y_sq2, top20_sq2_genes, "SQ2")
scp_relation = analyze_gene_label_relation_improved(X_scp_data, y_scp, top20_scp_genes, "SCP")

# 添加平台信息
sq2_relation['Platform'] = 'SQ2'
scp_relation['Platform'] = 'SCP'

# 合并结果
relation_all = pd.concat([sq2_relation, scp_relation], ignore_index=True)
relation_all.Fold_Change

relation_all.to_csv("logfoldchange show.csv")
# %%%  上下调可视化

# ================================
# 1. 数据准备和预处理
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16
sns.set_style("whitegrid")

# 加载数据（如果已经计算过）
# relation_all = pd.read_csv("logfoldchange_show.csv")
relation_all['abs_Fold_Change'] = relation_all['Fold_Change'].abs()
relation_all['-log10(p)'] = -np.log10(relation_all['ANOVA_p'] + 1e-10)  # 避免log(0)
relation_all['Significant'] = relation_all['ANOVA_p'] < 0.05
relation_all['Up_Down'] = np.where(relation_all['Fold_Change'] > 0, 'Up-regulated', 'Down-regulated')
relation_all['Expression_Pattern'] = relation_all['Max_Category'] + ' > ' + relation_all['Min_Category']

# 按平台分离
sq2_data = relation_all[relation_all['Platform'] == 'SQ2'].copy()
scp_data = relation_all[relation_all['Platform'] == 'SCP'].copy()

# 排序用于绘图
sq2_data = sq2_data.sort_values('Fold_Change', ascending=False)
scp_data = scp_data.sort_values('Fold_Change', ascending=False)

print("数据统计:")
print(f"SQ2平台: {len(sq2_data)} 个基因，{sq2_data['Significant'].sum()} 个显著")
print(f"SCP平台: {len(scp_data)} 个基因，{scp_data['Significant'].sum()} 个显著")
print("\n表达模式统计:")
print(relation_all.groupby(['Platform', 'Expression_Pattern']).size())

# ================================
# 2. 综合多图展示
# ================================
fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)

# ======================
# 2.1 平台比较条形图
# ======================
ax1 = fig.add_subplot(gs[0, 0])
platform_summary = relation_all.groupby('Platform').agg({
    'Gene': 'count',
    'Significant': 'sum',
    'abs_Fold_Change': 'mean'
}).reset_index()

x_pos = np.arange(len(platform_summary))
width = 0.3

bars1 = ax1.bar(x_pos - width/2, platform_summary['Gene'], 
                width, label='Total Genes', color='lightblue')
bars2 = ax1.bar(x_pos + width/2, platform_summary['Significant'], 
                width, label='Significant (p<0.05)', color='coral')

ax1.set_xlabel('Platform')
ax1.set_ylabel('Gene Count')
ax1.set_title('Gene Count by Platform')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(platform_summary['Platform'])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# ======================
# 2.2 Fold Change分布小提琴图
# ======================
ax2 = fig.add_subplot(gs[0, 1])
violin_parts = ax2.violinplot(
    [sq2_data['Fold_Change'], scp_data['Fold_Change']],
    showmeans=True, showmedians=True
)
ax2.set_xlabel('Platform')
ax2.set_ylabel('Fold Change')
ax2.set_title('Fold Change Distribution by Platform')
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['SQ2', 'SCP'])
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

# 着色
for i, pc in enumerate(violin_parts['bodies']):
    color = 'lightblue' if i == 0 else 'lightcoral'
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

# ======================
# 2.3 显著性水平比较
# ======================
ax3 = fig.add_subplot(gs[0, 2])
scatter = ax3.scatter(
    relation_all['Fold_Change'],
    relation_all['-log10(p)'],
    c=relation_all['Platform'].map({'SQ2': '#1f77b4', 'SCP': '#ff7f0e'}),
    s=relation_all['abs_Fold_Change'] * 100,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)
ax3.set_xlabel('Fold Change')
ax3.set_ylabel('-log10(p-value)')
ax3.set_title('Volcano-like Plot by Platform')
ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3)

# 标记显著基因
sig_genes = relation_all[relation_all['Significant']]
if not sig_genes.empty:
    for _, row in sig_genes.iterrows():
        ax3.annotate(row['Gene'], 
                    (row['Fold_Change'], row['-log10(p)']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)

# 添加图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='SQ2',
           markerfacecolor='#1f77b4', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='SCP',
           markerfacecolor='#ff7f0e', markersize=8),
    Line2D([0], [0], color='red', linestyle='--', label='p=0.05')
]
ax3.legend(handles=legend_elements, loc='best')

# ======================
# 2.4 SQ2平台基因瀑布图
# ======================
ax4 = fig.add_subplot(gs[1, :])
y_pos_sq2 = np.arange(len(sq2_data))
colors_sq2 = ['red' if x > 0 else 'blue' for x in sq2_data['Fold_Change']]
bars_sq2 = ax4.barh(y_pos_sq2, sq2_data['Fold_Change'], 
                    color=colors_sq2, alpha=0.7, edgecolor='black')

# 标记显著基因
for i, (bar, significant) in enumerate(zip(bars_sq2, sq2_data['Significant'])):
    if significant:
        bar.set_edgecolor('black')
        bar.set_linewidth(2)
        # 添加星号标记
        ax4.text(bar.get_width() + 0.02 * np.sign(bar.get_width()), 
                bar.get_y() + bar.get_height()/2, '*',
                ha='left' if bar.get_width() > 0 else 'right',
                va='center', fontsize=10, color='black', fontweight='bold')

ax4.set_yticks(y_pos_sq2)
ax4.set_yticklabels(sq2_data['Gene'])
ax4.set_xlabel('Fold Change')
ax4.set_title('SQ2 Platform: Gene Fold Changes (Red=Up, Blue=Down)')
ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
ax4.grid(axis='x', alpha=0.3)

# 添加表达模式标签
for i, pattern in enumerate(sq2_data['Expression_Pattern']):
    ax4.text(-0.1 if sq2_data.iloc[i]['Fold_Change'] < 0 else 0.02,
            i, pattern, ha='right' if sq2_data.iloc[i]['Fold_Change'] < 0 else 'left',
            va='center', fontsize=7, style='italic')

# ======================
# 2.5 SCP平台基因瀑布图
# ======================
ax5 = fig.add_subplot(gs[2, :])
y_pos_scp = np.arange(len(scp_data))
colors_scp = ['red' if x > 0 else 'blue' for x in scp_data['Fold_Change']]
bars_scp = ax5.barh(y_pos_scp, scp_data['Fold_Change'], 
                    color=colors_scp, alpha=0.7, edgecolor='black')

# 标记显著基因
for i, (bar, significant) in enumerate(zip(bars_scp, scp_data['Significant'])):
    if significant:
        bar.set_edgecolor('black')
        bar.set_linewidth(2)
        # 添加星号标记
        ax5.text(bar.get_width() + 0.02 * np.sign(bar.get_width()), 
                bar.get_y() + bar.get_height()/2, '*',
                ha='left' if bar.get_width() > 0 else 'right',
                va='center', fontsize=10, color='black', fontweight='bold')

ax5.set_yticks(y_pos_scp)
ax5.set_yticklabels(scp_data['Gene'])
ax5.set_xlabel('Fold Change')
ax5.set_title('SCP Platform: Gene Fold Changes (Red=Up, Blue=Down)')
ax5.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
ax5.grid(axis='x', alpha=0.3)

# 添加表达模式标签
for i, pattern in enumerate(scp_data['Expression_Pattern']):
    ax5.text(-0.1 if scp_data.iloc[i]['Fold_Change'] < 0 else 0.02,
            i, pattern, ha='right' if scp_data.iloc[i]['Fold_Change'] < 0 else 'left',
            va='center', fontsize=7, style='italic')

# ======================
# 2.6 表达模式热图
# ======================
ax6 = fig.add_subplot(gs[3, 0])
pattern_counts = relation_all.groupby(['Platform', 'Expression_Pattern']).size().unstack(fill_value=0)

# 如果模式太多，只显示前几种
if pattern_counts.shape[1] > 8:
    pattern_counts = pattern_counts[pattern_counts.sum().nlargest(8).index]

pattern_counts.plot(kind='bar', stacked=True, ax=ax6, colormap='Set3')
ax6.set_xlabel('Platform')
ax6.set_ylabel('Gene Count')
ax6.set_title('Expression Patterns by Platform')
ax6.legend(title='Pattern', bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.tick_params(axis='x', rotation=0)

# ======================
# 2.7 Fold Change vs F-statistic
# ======================
ax7 = fig.add_subplot(gs[3, 1])
scatter = ax7.scatter(
    relation_all['Fold_Change'],
    relation_all['ANOVA_F'],
    c=relation_all['Platform'].map({'SQ2': '#1f77b4', 'SCP': '#ff7f0e'}),
    s=relation_all['-log10(p)'] * 20,
    alpha=0.7,
    edgecolors='black'
)
ax7.set_xlabel('Fold Change')
ax7.set_ylabel('ANOVA F-statistic')
ax7.set_title('Fold Change vs Statistical Power')
ax7.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax7.grid(True, alpha=0.3)

# 标记top基因
top_n = 5
top_genes = relation_all.nlargest(top_n, 'ANOVA_F')
for _, row in top_genes.iterrows():
    ax7.annotate(row['Gene'], 
                (row['Fold_Change'], row['ANOVA_F']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold')

# ======================
# 2.8 平台间差异箱线图
# ======================
ax8 = fig.add_subplot(gs[3, 2])
platform_data = [sq2_data['Fold_Change'], scp_data['Fold_Change']]
bp = ax8.boxplot(platform_data, patch_artist=True, widths=0.6)

# 设置颜色
colors = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax8.set_xticklabels(['SQ2', 'SCP'])
ax8.set_ylabel('Fold Change')
ax8.set_title('Platform Comparison of Fold Changes')
ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax8.grid(axis='y', alpha=0.3)

# 添加统计检验
if len(sq2_data) > 0 and len(scp_data) > 0:
    t_stat, p_val = stats.ttest_ind(sq2_data['Fold_Change'], scp_data['Fold_Change'], 
                                    equal_var=False)
    ax8.text(0.5, 0.95, f'p = {p_val:.3f}', 
             transform=ax8.transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ================================
# 3. 保存图像
# ================================
plt.suptitle('Gene Expression Analysis by Calcium Response and Platform', 
             fontsize=16, y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig('gene_expression_analysis_comprehensive.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('gene_expression_analysis_comprehensive.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# ================================
# 4. 单独的热图可视化
# ================================
# 创建基因表达热图
fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(16, 10))

# 4.1 SQ2平台热图
if len(sq2_data) > 0:
    sq2_heatmap_data = pd.DataFrame({
        'Fold Change': sq2_data['Fold_Change'],
        '-log10(p)': sq2_data['-log10(p)'],
        'ANOVA F': sq2_data['ANOVA_F']
    }, index=sq2_data['Gene']).T
    
    im1 = ax21.imshow(sq2_heatmap_data, cmap='RdBu_r', aspect='auto', 
                     interpolation='nearest')
    ax21.set_xticks(range(len(sq2_data)))
    ax21.set_xticklabels(sq2_data['Gene'], rotation=45, ha='right', fontsize=9)
    ax21.set_yticks(range(len(sq2_heatmap_data.index)))
    ax21.set_yticklabels(sq2_heatmap_data.index)
    ax21.set_title('SQ2 Platform: Gene Expression Metrics Heatmap')
    plt.colorbar(im1, ax=ax21, fraction=0.046, pad=0.04)

# 4.2 SCP平台热图
if len(scp_data) > 0:
    scp_heatmap_data = pd.DataFrame({
        'Fold Change': scp_data['Fold_Change'],
        '-log10(p)': scp_data['-log10(p)'],
        'ANOVA F': scp_data['ANOVA_F']
    }, index=scp_data['Gene']).T
    
    im2 = ax22.imshow(scp_heatmap_data, cmap='RdBu_r', aspect='auto',
                     interpolation='nearest')
    ax22.set_xticks(range(len(scp_data)))
    ax22.set_xticklabels(scp_data['Gene'], rotation=45, ha='right', fontsize=9)
    ax22.set_yticks(range(len(scp_heatmap_data.index)))
    ax22.set_yticklabels(scp_heatmap_data.index)
    ax22.set_title('SCP Platform: Gene Expression Metrics Heatmap')
    plt.colorbar(im2, ax=ax22, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('gene_expression_heatmaps.pdf', dpi=300, bbox_inches='tight')
plt.savefig('gene_expression_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# ================================
# 5. 气泡图：三变量可视化
# ================================
fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(16, 8))

# 5.1 SQ2平台气泡图
if len(sq2_data) > 0:
    scatter1 = ax31.scatter(
        sq2_data['Fold_Change'],
        range(len(sq2_data)),
        s=sq2_data['-log10(p)'] * 100,
        c=sq2_data['ANOVA_F'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='black'
    )
    ax31.set_yticks(range(len(sq2_data)))
    ax31.set_yticklabels(sq2_data['Gene'])
    ax31.set_xlabel('Fold Change')
    ax31.set_title('SQ2: Fold Change vs Significance (Size=-log10(p), Color=F-stat)')
    ax31.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.colorbar(scatter1, ax=ax31, label='ANOVA F-statistic')

# 5.2 SCP平台气泡图
if len(scp_data) > 0:
    scatter2 = ax32.scatter(
        scp_data['Fold_Change'],
        range(len(scp_data)),
        s=scp_data['-log10(p)'] * 100,
        c=scp_data['ANOVA_F'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='black'
    )
    ax32.set_yticks(range(len(scp_data)))
    ax32.set_yticklabels(scp_data['Gene'])
    ax32.set_xlabel('Fold Change')
    ax32.set_title('SCP: Fold Change vs Significance (Size=-log10(p), Color=F-stat)')
    ax32.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.colorbar(scatter2, ax=ax32, label='ANOVA F-statistic')

plt.tight_layout()
plt.savefig('gene_expression_bubble_charts.pdf', dpi=300, bbox_inches='tight')
plt.savefig('gene_expression_bubble_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# ================================
# 6. 生成分析报告
# ================================
print("\n" + "="*50)
print("分析报告")
print("="*50)

# 6.1 平台特异的关键基因
print("\n1. 平台特异的关键显著基因:")
for platform in ['SQ2', 'SCP']:
    platform_data = relation_all[relation_all['Platform'] == platform]
    sig_genes = platform_data[platform_data['Significant']]
    
    if len(sig_genes) > 0:
        print(f"\n{platform}平台显著基因 ({len(sig_genes)}个):")
        for _, row in sig_genes.iterrows():
            direction = "上调" if row['Fold_Change'] > 0 else "下调"
            print(f"  - {row['Gene']}: {direction} (FC={row['Fold_Change']:.2f}, "
                  f"p={row['ANOVA_p']:.3f}, F={row['ANOVA_F']:.1f})")
            print(f"    表达模式: {row['Max_Category']} > {row['Min_Category']}")

# 6.2 平台间对比
print("\n2. 平台间对比:")
print(f"平均Fold Change:")
print(f"  SQ2: {sq2_data['Fold_Change'].mean():.3f}")
print(f"  SCP: {scp_data['Fold_Change'].mean():.3f}")

print(f"\n显著性基因比例:")
print(f"  SQ2: {sq2_data['Significant'].mean():.1%}")
print(f"  SCP: {scp_data['Significant'].mean():.1%}")

# 6.3 主要表达模式
print("\n3. 主要表达模式:")
common_patterns = relation_all.groupby('Expression_Pattern').agg({
    'Gene': 'count',
    'Fold_Change': 'mean',
    'ANOVA_p': lambda x: (x < 0.05).sum()
}).sort_values('Gene', ascending=False).head(5)

print(common_patterns)

# 6.4 保存详细数据
print("\n4. 数据保存:")
relation_all_sorted = relation_all.sort_values(['Platform', '-log10(p)'], 
                                               ascending=[True, False])
relation_all_sorted.to_csv('gene_expression_analysis_detailed.csv', index=False)
print("详细数据已保存: gene_expression_analysis_detailed.csv")

summary_stats = relation_all.groupby('Platform').agg({
    'Gene': 'count',
    'Fold_Change': ['mean', 'std', 'min', 'max'],
    'ANOVA_p': lambda x: (x < 0.05).sum()
}).round(3)
summary_stats.to_csv('gene_expression_summary_stats.csv')
print("统计摘要已保存: gene_expression_summary_stats.csv")

print("\n" + "="*50)
print("可视化完成！")
print(f"生成文件:")
print("1. gene_expression_analysis_comprehensive.pdf/png - 综合图")
print("2. gene_expression_heatmaps.pdf/png - 热图")
print("3. gene_expression_bubble_charts.pdf/png - 气泡图")
print("4. gene_expression_analysis_detailed.csv - 详细数据")
print("5. gene_expression_summary_stats.csv - 统计摘要")
print("="*50)

# %% 基因上下调
# 用和建模一致的表达矩阵
X_expr = X_scaled[common_genes]   # cells × genes
platform_label = X_scaled["platform_SQ2"].map({True: "SQ2", False: "SCP"})

response_label = y
def calc_up_down(X_expr, platform_label, response_label, platform):
    idx = platform_label == platform
    
    X_p = X_expr.loc[idx]
    y_p = response_label.loc[idx]
    
    mean_resp = X_p.loc[y_p == "Responsive"].mean(axis=0)
    mean_non  = X_p.loc[y_p == "Non-Responsive"].mean(axis=0)
    
    delta = mean_resp - mean_non
    
    return pd.DataFrame({
        "gene": delta.index,
        "delta_expr": delta.values,
        "direction": np.where(delta > 0, "Up", "Down")
    }).set_index("gene")

sq2_ud = calc_up_down(X_expr, platform_label, response_label, "SQ2")
scp_ud = calc_up_down(X_expr, platform_label, response_label, "SCP")
sq2_plot_df = pd.DataFrame({
    "gene": top20_sq2.index,
    "MDA": top20_sq2.values
}).set_index("gene").join(sq2_ud)

sq2_plot_df["platform"] = "SQ2"


scp_plot_df = pd.DataFrame({
    "gene": top20_scp.index,
    "MDA": top20_scp.values
}).set_index("gene").join(scp_ud)

scp_plot_df["platform"] = "SCP"
plot_df = pd.concat([sq2_plot_df, scp_plot_df])


# %%% 谱图

import matplotlib.pyplot as plt

marker_map = {
    "SQ2": "o",
    "SCP": "^"
}

color_map = {
    "Up": "red",
    "Down": "blue"
}

plt.figure(figsize=(8, 10))

# 为了让重要基因在上面
plot_df_sorted = plot_df.sort_values("MDA")

y_pos = range(len(plot_df_sorted))

for i, (gene, row) in enumerate(plot_df_sorted.iterrows()):
    plt.scatter(
        row["MDA"],
        i,
        s=row["MDA"] * 4500,
        marker=marker_map[row["platform"]],
        color=color_map[row["direction"]],
        edgecolors="black",
        alpha=0.8
    )

plt.yticks(y_pos, plot_df_sorted.index)
plt.xlabel("Permutation Importance (MDA)")
plt.title("Platform-specific Up/Down-regulated Gene Importance")

# 图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='SQ2',
           markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='SCP',
           markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Up',
           markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Down',
           markerfacecolor='blue', markersize=10)
]

plt.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()

plt.savefig("Platform_UpDown_MDA_Bubble.pdf", bbox_inches="tight")
plt.savefig("Platform_UpDown_MDA_Bubble.svg", bbox_inches="tight")
plt.show()


# %%fix



X_expr = X_scaled[common_genes]

platform_label = X_scaled["platform_SQ2"].map({
    True: "SQ2",
    False: "SCP"
})

response_label = y  # Responsive / Weak / Non-Responsive
def calc_expr_signature(X_expr, platform_label, response_label, platform):
    idx_p = platform_label == platform
    X_p = X_expr.loc[idx_p]
    y_p = response_label.loc[idx_p]

    # 以 Non-Responsive 为基线
    baseline = X_p.loc[y_p == "Non-Responsive"].mean(axis=0)

    records = []
    for state in ["Responsive", "Weak", "Non-Responsive"]:
        if (y_p == state).sum() == 0:
            continue

        mean_state = X_p.loc[y_p == state].mean(axis=0)
        delta = mean_state - baseline

        df = pd.DataFrame({
            "gene": delta.index,
            "delta_expr": delta.values,
            "state": state,
            "platform": platform,
            "direction": np.where(delta > 0, "Up", "Down")
        })
        records.append(df)

    return pd.concat(records)
sig_sq2 = calc_expr_signature(
    X_expr, platform_label, response_label, "SQ2"
)

sig_scp = calc_expr_signature(
    X_expr, platform_label, response_label, "SCP"
)

plot_df = pd.concat([sig_sq2, sig_scp])

# %%% plot
# 分别获取各平台 top genes
top_sq2_genes = list(top20_sq2.index)
top_scp_genes = list(top20_scp.index)

# 平台对应筛选
sig_sq2_top = sig_sq2[sig_sq2['gene'].isin(top_sq2_genes)]
sig_scp_top = sig_scp[sig_scp['gene'].isin(top_scp_genes)]

# 合并绘图数据
plot_df_filtered = pd.concat([sig_sq2_top, sig_scp_top])

# 设置 x_label 顺序
x_order = [
    "SQ2_Responsive",
    "SQ2_Weak",
    "SQ2_Non-Responsive",
    "SCP_Responsive",
    "SCP_Weak",
    "SCP_Non-Responsive",
]
plot_df_filtered["x_label"] = (
    plot_df_filtered["platform"] + "_" + plot_df_filtered["state"]
)
plot_df_filtered["x_label"] = pd.Categorical(
    plot_df_filtered["x_label"],
    categories=x_order,
    ordered=True
)

# 绘图
plt.figure(figsize=(15, 0.35 * len(plot_df_filtered["gene"].unique())))

genes = plot_df_filtered["gene"].unique()
gene_to_y = {g: i for i, g in enumerate(genes)}

for _, row in plot_df_filtered.iterrows():
    plt.scatter(
        row["x_label"],
        gene_to_y[row["gene"]],
        s=200,
        marker=marker_map[row["platform"]],
        color=color_map[row["direction"]],
        edgecolors="black",
        alpha=0.85
    )

plt.yticks(range(len(genes)), genes)
plt.xlabel("Platform × Calcium Response")
plt.ylabel("Gene")
plt.title("Top Genes per Platform: Expression Signatures")

# 图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='SQ2',
           markerfacecolor='gray', markersize=9),
    Line2D([0], [0], marker='^', color='w', label='SCP',
           markerfacecolor='gray', markersize=9),
    Line2D([0], [0], marker='o', color='w', label='Up-regulated',
           markerfacecolor='red', markersize=9),
    Line2D([0], [0], marker='o', color='w', label='Down-regulated',
           markerfacecolor='blue', markersize=9),
]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig("Platform_Response_TopGenes_Bubble.pdf", bbox_inches="tight")
plt.savefig("Platform_Response_TopGenes_Bubble.svg", bbox_inches="tight")
plt.show()
