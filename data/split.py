import pandas as pd

df = pd.read_csv('train.csv')

# 将df划分为训练和测试集
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.2)
# 保存数据集
train_data.to_csv('train-train.csv', index=False)
test_data.to_csv('train-valid.csv', index=False)
