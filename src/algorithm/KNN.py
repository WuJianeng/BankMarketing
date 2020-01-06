from pandas import read_csv
import pandas
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

filename = '../data/bank-full.csv'
data = read_csv(filename, delimiter=";")

# preProcessing
data['job'] = LabelEncoder().fit_transform(data['job'])
data['marital'] = LabelEncoder().fit_transform(data['marital'])
data['education'] = LabelEncoder().fit_transform(data['education'])
data['default'] = LabelEncoder().fit_transform(data['default'])
data['housing'] = LabelEncoder().fit_transform(data['housing'])
data['loan'] = LabelEncoder().fit_transform(data['loan'])
data['contact'] = LabelEncoder().fit_transform(data['contact'])
data['month'] = LabelEncoder().fit_transform(data['month'])
data['poutcome'] = LabelEncoder().fit_transform(data['poutcome'])
data['y'] = LabelEncoder().fit_transform(data['y'])

# for i in range(40000):
#     if data[i][16] == 0:
#         data.drop([i])

# data[['y']] = data[['y']].replace(0, numpy.NAN)
# data.dropna(axis=0, how='any', inplace=True)
data.drop(labels=["day", "month"], axis=1, inplace=True)
print(data.head(5))

array = data.values
X = array[:, 0:14]
print(X)
Y = array[:, 14]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)

print("x_train\n" + str(x_train))
print("x_test\n" + str(x_test))
print("y_test\n" + str(y_test))

# KNN
model = KNeighborsClassifier()
model.fit(x_train, y_train)
print("KNN准确率: " + str(model.score(x_test, y_test)))

# SVM
# model = SVC()
# model.fit(x_train, y_train)
# print("SVC准确率: " + str(model.score(x_test, y_test)))

# LogisticRegression
# model = LogisticRegression()
# model.fit(x_train, y_train)
# print("LogisticRegression准确率: " + str(model.score(x_test, y_test)))

# RandomForest
model = RandomForestClassifier()
model.fit(x_train, y_train)
print("Random Forest score: " + str(model.score(x_test, y_test)))

# DecisionTree
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("DecisionTree score: " + str(model.score(x_test, y_test)))

num = 0.0
for y in y_test:
    if y == 0:
        num += 1

res = num / len(y_test)
print("0的比例: " + str(res))

