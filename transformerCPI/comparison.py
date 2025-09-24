import pandas as pd

# 读取文件
df_official = pd.read_csv("result_official1.csv")
df_999 = pd.read_csv("result_995.csv")

# 确保两者都有同样的分子标识列（假设为 'smiles'）
key_column = 'smiles'  # 也可以改为 'SMILES' 或 'id'，视具体列名而定

# 合并，使用内连接只保留出现在 df_999 中的分子
merged = pd.merge(df_999[[key_column, 'prediction']], df_official, on=key_column, how='inner')

# 按预测值降序排序
merged_sorted = merged.sort_values(by='prediction', ascending=False)

# 输出结果
merged_sorted.to_csv("final.csv", index=False)
print(f"✅ 筛选并排序完成，输出 {len(merged_sorted)} 条记录至 final.csv")