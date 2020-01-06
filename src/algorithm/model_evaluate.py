from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from algorithm import data_func
from algorithm import dataset
import numpy as np
from matplotlib import pylab


def plot_pr(auc_score, precision, recall, label=None):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    pylab.fill_between(recall, precision, alpha=0.2)
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(recall, precision, lw=1)
    pylab.show()


def plot_roc(auc_score, fpr, tpr, label=None):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False positive rate')
    pylab.ylabel('True positive rate')
    pylab.title('ROC (AUC=%0.2f) / %s' % (auc_score, label))
    pylab.fill_between(fpr, tpr, alpha=0.2)
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(fpr, tpr, lw=1)
    pylab.show()


def train_evaluate(train_data, test_data, model, n=1, frac=1.0, threshold=0.5):
    train_data = dataset.re_sample(train_data, n, frac)
    train_x, train_y = train_data.drop('y', axis=1), train_data['y']
    test_x, test_y = test_data.drop('y', axis=1), test_data['y']

    # model = RandomForestClassifier()
    model.fit(train_x, train_y)
    predict_y_proba = model.predict_proba(test_x)[:, 1]
    report = classification_report(test_y, predict_y_proba > threshold,
                                   target_names=['no', 'yes'])
    predict_y = (predict_y_proba > threshold).astype(int)
    accuracy = np.mean(test_y.values == predict_y)
    print("Accuracy: {}".format(accuracy))
    print(report)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y_proba)
    precision, recall, thresholds = metrics.precision_recall_curve(test_y, predict_y_proba)
    test_auc = metrics.auc(fpr, tpr)
    plot_pr(test_auc, precision, recall, 'yes')
    print("end...")
    return predict_y


if __name__ == '__main__':
    data = data_func.read_processed_data()
    train_data, test_data = dataset.split_data_set(data)
    # model = RandomForestClassifier()
    model = DecisionTreeClassifier()
    # model = LogisticRegression()
    # model = KNeighborsClassifier()
    train_evaluate(train_data, test_data, model, n=8, frac=1, threshold=0.5)
