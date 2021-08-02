import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = (10,6)
plt.style.use('ggplot')
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
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

# Read datasets from csv files:

bear_class = pd.read_csv('bearing_classes.csv', sep=',', index_col=0)
bear_signal = pd.read_csv('bearing_signals.csv', sep=',', index_col=0)


# Separate Limit-Test Experiments from Vibration-Test Experiments:

limit_test = bear_signal.copy()
vibration_test = bear_signal.copy()

limit_exp = [8,11,14,15,17,19,21,23,24,29,36,81]

for num in range(2,113):
    if num not in limit_exp:
        limit_test.drop(limit_test[limit_test['experiment_id'] == num].index, inplace=True)

for num in limit_exp:
    vibration_test.drop(vibration_test[vibration_test['experiment_id'] == num].index, inplace=True)


# New Dataset: Slice Vibration-Test Experiments dataset on condition: "Timestamp <= 1.5 sec":

low_speed_set = vibration_test[vibration_test['timestamp'] <= 1.5]


# Write new datasets to csv files:

limit_test.to_csv('limit_test.csv')
vibration_test.to_csv('vibration_test.csv')
low_speed_set.to_csv('low_speed_set.csv')


