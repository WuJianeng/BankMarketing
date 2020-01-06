from pandas import read_csv
import os


def read_data():
    filename = 'D:/MachineLearning/BankMarketing/bank/bank-full.csv'
    data = read_csv(filename, delimiter=";")
    return data


def read_processed_data():
    filename = 'D:/MachineLearning/BankMarketing/bank/processed_data.csv'
    data = read_csv(filename)
    return data


def arrange_arrays():
    arrays = []
    bin_attrs = ['default', 'housing', 'loan']
    cate_attrs = ['job', 'marital', 'contact', 'month', 'poutcome']
    numeric_attrs = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    return bin_attrs, cate_attrs, numeric_attrs


def write_to_csv(data, filename):
    filepath = 'D:/MachineLearning/BankMarketing/bank/' + filename
    data.to_csv(filepath, index=False)


if __name__ == '__main__':
    # data_new = read_data()
    # print(data_new.info)
    # bin, cate, numeric = arrange_arrays()
    # print(bin, cate, numeric)
    # write_to_csv(data_new, 'temp.csv')
    data_new = read_processed_data()
    print(data_new.info())
    print("   ---------      ")
    print(data_new.head(10))
