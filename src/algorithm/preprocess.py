import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from algorithm import data_func


def process_data(data, bin_attrs, cate_attrs, numeric_attrs):
    # print("delete pre: " + str(data['y'].count()))
    fill_attrs = []
    attrs = []
    zeros = []
    for i in data.columns:
        if 500 > data[data[i] == 'unknown']['y'].count() > 0:
            data = data[data[i] != 'unknown']
            attrs.append(i)
        elif data[data[i] == 'unknown']['y'].count() > 500:
            fill_attrs.append(i)
        else:
            zeros.append(i)
    # print("500plus: " + fill_attrs.__str__())
    # print("done: " + attrs.__str__())
    # print("zeros: " + zeros.__str__())
    # print("deleted: " + str(data['y'].count()))
    data = encode_bin_attrs(data, bin_attrs)
    data = encode_edu_attrs(data)
    data = encode_cate_attrs(data, cate_attrs)
    # y 编码
    data = encode_y_attr(data)
    # 数值特征归一
    feature_scaling(data, numeric_attrs)
    # 缺失值填充
    fill_attrs.remove('contact')
    fill_attrs.remove('poutcome')
    # print("fill_attrs: " + str(fill_attrs))
    data = shuffle(data)
    data = fill_unknown(data, fill_attrs)

    data['y'] = data['y'].astype(int)
    # pd.set_option('display.width', None)
    # print(data.head(10))
    return data


def encode_y_attr(data):
    attr = 'y'
    data.loc[data[attr] == 'yes', attr] = 1
    data.loc[data[attr] == 'no', attr] = 0
    return data


# 二元变量直接二元编码
def encode_bin_attrs(data, bin_attrs):
    for i in bin_attrs:
        data.loc[data[i] == 'yes', i] = 1
        data.loc[data[i] == 'no', i] = 0
    return data


# education 数值化
def encode_edu_attrs(data):
    attr = 'education'
    data.loc[data[attr] == 'secondary', attr] = 1
    data.loc[data[attr] == 'primary', attr] = 2
    data.loc[data[attr] == 'tertiary', attr] = 3
    return data


def encode_cate_attrs(data, cate_attrs):
    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i + '_' + str(x))
        data = pd.concat([data, dummies_df], axis=1)
        data = data.drop(i, axis=1)
    return data


def train_predict_unknown(train_x, train_y, test_x):
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_x, train_y.astype(int))
    print(test_x.count())
    test_predict_y = forest.predict(test_x).astype(int)
    return pd.DataFrame(test_predict_y, index=test_x.index)


# 缺失值处理
def fill_unknown(data, fill_attrs):
    for i in fill_attrs:
        test_data = data[data[i] == 'unknown']
        test_x = test_data.drop(fill_attrs, axis=1)
        train_data = data[data[i] != 'unknown']
        train_y = train_data[i]
        train_x = train_data.drop(fill_attrs, axis=1)
        test_data[i] = train_predict_unknown(train_x, train_y, test_x)
        data = pd.concat([train_data, test_data])
        data[i] = data[i].astype(int)
    return data


def feature_scaling(data, numeric_attrs):
    for i in numeric_attrs:
        # scaler = preprocessing.StandardScaler()
        # data[i] = scaler.fit_transform(data[i])
        std = data[i].std()
        # print("\nstd: " + str(std) + "\nmean: " + str(mean))
        if std != 0:
            mean = data[i].mean()
            data[i] = (data[i] - mean) / std
        else:
            data = data.drop(i, axis=1)
    return data


if __name__ == '__main__':
    data_file = data_func.read_data()
    # pd.set_option('display.width', None)
    # print(data_file.head(20))
    # print(data_file.count())
    # print(data_file[data_file['contact'] == 'unknown'].count())
    bin_attrs, cate_attrs, numeric_attrs_actual = data_func.arrange_arrays()
    result = process_data(data_file, bin_attrs, cate_attrs, numeric_attrs_actual)
    data_func.write_to_csv(result, "processed_data.csv")
    print(result.head(10))
