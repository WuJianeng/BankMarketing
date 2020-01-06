from pandas import read_csv
import pandas
import numpy
from matplotlib import pyplot as plt
import seaborn

filename = '../data/bank-full.csv'
data = read_csv(filename, delimiter=";")
names = ['age', 'job', 'marital', 'education', 'default', 'balance','housing', 'loan', 'contact', 'day'
         'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
# 数据信息 no-null
pandas.set_option('display.width', None)
print(data.info())
# 数据描述
print(data.describe())
# 数据分布
data.hist(bins=40, figsize=(14, 10))
plt.show()

# 职业分布
# plt.rcParams['figure.figsize'] = (10, 6)
# seaborn.set()
# seaborn.barplot(x='index', y='job', data=data['job'].value_counts().to_frame().reset_index())
# plt.xticks(rotation=90)
# plt.show()

# 职业与年龄
# seaborn.boxplot(x='job', y='age', data=data)
# plt.xticks(rotation=90)
# plt.show()

# 职业与收入分布情况
# seaborn.boxplot(x='job', y='balance', data=data)
# plt.xticks(rotation=90)
# plt.show()

# 受教育情况
# plt.rcParams['figure.figsize'] = (10, 6)
# seaborn.set()
# seaborn.barplot(x='index', y='education', data=data['education'].value_counts().to_frame().reset_index())
# plt.xticks(rotation=90)
# plt.show()

# 受教育情况和年龄分布
# seaborn.boxplot(x='education', y='age', data=data)
# plt.xticks(rotation=90)
# plt.show()

# 受教育和收=收入分布情况
# seaborn.boxplot(x='education', y='balance', data=data)
# plt.xticks(rotation=90)
# plt.show()

print(data.head(10))
