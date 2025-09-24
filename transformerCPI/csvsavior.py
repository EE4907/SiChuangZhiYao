import pandas as pd
import re

# 读取 CSV 文件
df = pd.read_csv('final copy.csv')  # 如果没有标题行，加 header=None

# 定义提取函数
def extract_title_smiles(cell):
    match = re.search(r'^\d+.*?[A-Za-z]', cell)
    if match:
        # 找到第一个字母的位置
        first_alpha_index = re.search(r'[A-Za-z]', cell).start()
        title = cell[:first_alpha_index]
        smiles = cell[first_alpha_index:]
        return pd.Series([title.strip(), smiles.strip()])
    else:
        return pd.Series([cell.strip(), ''])

# 应用提取函数
df[['Catalog_No', 'SMILES']] = df[0].apply(extract_title_smiles)

# 显示结果
print(df[['Catalog_No', 'SMILES']])