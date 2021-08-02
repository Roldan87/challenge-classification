# Add you code here.
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
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


# Read dataset:

low_speed = read_csv('low_speed_set.csv', index_col=0)


# Replace old values for mean values per experiment
mean_set = low_speed.copy()




# Merge Target column to dataset:




# RandomForestClassifier Model:

features = ['timestamp', 'a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z', 'rpm']
X = pd.get_dummies(mean_set[features])
y = mean_set['target'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

rfc = RandomForestClassifier().fit(X_train, y_train)

y_pred = rfc.predict(X_test)

cv_scores = cross_val_score(rfc, X_train, y_train, cv=5)
score = accuracy_score(y_test, y_pred)

# Print Results:

print(cv_scores)
print(score)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 1.0
# 1.0
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00    117978
#            1       1.00      1.00      1.00     16926
# 
#     accuracy                           1.00    134904
#    macro avg       1.00      1.00      1.00    134904
# weighted avg       1.00      1.00      1.00    134904
# 
# [[117978      0]
#  [     0  16926]]