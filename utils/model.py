# Add you code here.
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

pd.set_option('display.max_columns', None)


def fit_evaluate_model_random_forest(df, features_to_exclude, test_size):
    X = df.drop(features_to_exclude, axis=1)
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)

    #scaling? Surely with rpm, high values
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    rfc = RandomForestClassifier().fit(X_train, y_train)
    cv_scores = cross_val_score(rfc, X_train, y_train, cv=5)
    y_pred = rfc.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print('cross validation score:', cv_scores)
    print('score:', score)
    print('classification_report:\n', classification_report(y_test, y_pred))
    print('confusion_matrix:\n', confusion_matrix(y_test, y_pred))
    print('roc_auc_score', roc_auc_score(y_test, y_pred))
    plot_roc_curve(rfc, X_test, y_test)
    plt.show()

    # [1. 1. 1. 1. 1.]
    # 1.0
    # precision    recall  f1-score   support
    #
    #            0       1.00      1.00      1.00    117248
    #            1       1.00      1.00      1.00     17782
    #
    #     accuracy                           1.00    135030
    #    macro avg       1.00      1.00      1.00    135030
    # weighted avg       1.00      1.00      1.00    135030

    # [[117248      0]
    #  [     0  17782]]
