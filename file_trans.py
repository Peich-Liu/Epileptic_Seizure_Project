import pandas as pd
# 读取Parquet文件
df = pd.read_parquet('/home/pliu/git_repo/10_datasets/SIENA_Standardized/PN00/PN00-1.edf')

# 将DataFrame写入CSV文件
df.to_csv('13.csv', index=False)