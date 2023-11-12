import pandas as pd
# 读取Parquet文件
df = pd.read_parquet('/home/pliu/git_repo/10_datasets/SIENA_Features/PN00/PN00-1-DMe.parquet.gzip')

# 将DataFrame写入CSV文件
df.to_csv('net.csv', index=False)