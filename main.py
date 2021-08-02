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


# New Dataset: Slice Vibration-Test Experiments dataset on condition: "Timestamp <= 1.5 sec":

low_speed_set = bear_signal[bear_signal['timestamp'] >= 0.1620]
low_speed_set = bear_signal[bear_signal['timestamp'] <= 1.5]


# Write new datasets to csv files:

low_speed_set.to_csv('low_speed_set.csv')


