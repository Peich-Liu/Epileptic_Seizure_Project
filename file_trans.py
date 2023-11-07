import pandas as pd
# 读取Parquet文件
df = pd.read_parquet('/home/pliu/git_repo/10_datasets/SIENA_multi_Features_/PN01/PN01-1-StandardDeviation.parquet.gzip')

# 将DataFrame写入CSV文件
df.to_csv('01.csv', index=False)