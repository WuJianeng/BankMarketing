from pandas import read_csv
import pandas
import numpy
from matplotlib import pyplot as plt
import seaborn
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

filename = '../data/bank-full.csv'
data = read_csv(filename, delimiter=";")

for i in data.columns:
    if data[data[i] == 'unknown']['y'].count() > 0:
        print(i)

# 存款转变为数值变量
# data['y'] = LabelEncoder().fit_transform(data['y'])
# print(data['y'])
# corrmat = data.corr()
# print(corrmat)
# plt.figure(figsize=(15, 10))
# # seaborn.heatmap(corrmat, annot=True, cmap=seaborn.diverging_palette(220, 20, as_camp=True))
# seaborn.heatmap(corrmat, annot=True, cmap="Blues", vmax=1, square=True)
# plt.yticks(rotation=90)
# plt.show()

pandas.set_option('display.width', None)
data_yes = data[data['y'] == 'yes']
print(data_yes.describe())

data_no = data[data['y'] == 'no']
print(data_no.describe())
print(data_no.info())

# data_yes.marital.value_counts().plot(kind='pie',autopct='%1.1f%%')
# plt.show()
data_no.marital.value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
