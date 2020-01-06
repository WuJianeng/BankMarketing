from sklearn.ensemble import RandomForestClassifier
from algorithm import preprocess
from algorithm import data_func
from algorithm import dataset


data = data_func.read_processed_data()
train_data, test_data = dataset.split_data_set(data)
train_data = dataset.re_sample(train_data, n=8, fraction=1)
train_data_x, train_data_y = train_data.drop('y', axis=1), train_data['y']

test_data = test_data[test_data['y'] == 1]
test_data_x, test_data_y = test_data.drop('y', axis=1), test_data['y']
model = RandomForestClassifier()
model.fit(train_data_x, train_data_y)
print(model.score(test_data_x, test_data_y))
