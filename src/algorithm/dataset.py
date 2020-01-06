from sklearn.model_selection import train_test_split
from algorithm import data_func
import pandas as pd
from sklearn.utils import shuffle
from algorithm import smote


def split_data_set(data):
    # x, y = data.drop('y', axis=1), data['y']
    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=7)
    # return train_x, train_y, test_x, test_y
    data_len = data['y'].count()
    train_len = int(data_len * 0.8)
    train_data = data[:train_len]
    test_data = data[train_len:]
    return train_data, test_data


def re_sample(train_data, n, fraction):
    numeric_attrs = ['age', 'duration', 'campaign', 'pdays', 'previous']

    pos_train_data_original = train_data[train_data['y'] == 1]
    pos_train_data = train_data[train_data['y'] == 1]
    new_count = n * pos_train_data['y'].count()
    neg_train_data = train_data[train_data['y'] == 0].sample(frac=fraction)
    train_list = []
    if n != 0:
        pos_train_x = pos_train_data[numeric_attrs]
        pos_train_x2 = pd.concat([pos_train_data.drop(numeric_attrs, axis=1)] * n)
        pos_train_x2.index = range(new_count)

        s = smote.Smote(pos_train_x.values, N=n, k=3)
        pos_train_x = s.over_sampling()
        pos_train_x = pd.DataFrame(pos_train_x, columns=numeric_attrs, index=range(new_count))

        pos_train_data = pd.concat([pos_train_x, pos_train_x2], axis=1)
        pos_train_data = pd.DataFrame(pos_train_data, columns=pos_train_data_original.columns)
        train_list = (pos_train_data, neg_train_data, pos_train_data_original)
    else:
        train_list = [neg_train_data, pos_train_data_original]
    print("Size of positive train data: {} * {}".format(pos_train_data_original['y'].count(), n+1))
    print("Size of negtive train data: {} * {}".format(neg_train_data['y'].count(), fraction))
    train_data = pd.concat(train_list, axis=0)
    return shuffle(train_data)


if __name__ == '__main__':
    data = data_func.read_processed_data()
    train_data, test_data = split_data_set(data)
    train_data = re_sample(train_data, n=8, fraction=1)
    print("yes's count: {}".format(train_data[train_data['y'] == 1]['y'].count()))
    print("all's count: {}".format(train_data['y'].count()))
