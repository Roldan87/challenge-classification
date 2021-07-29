import pandas as pd

# Reading ONLY 100k rows.
features = pd.read_csv("bearing_signals.csv", nrows=100000)
target = pd.read_csv("bearing_classes.csv", sep=";")

# Renaming id-columns
# Dropping redundant "rpm"-column
features = features.rename(columns={"bearing_1_id":"control_bearing","bearing_2_id":"test_bearing"})
features = features.drop(columns=["rpm"], axis=1)
print(features.columns)
print(features.test_bearing.value_counts())

# Making two DF's: control and test group
control_bearing = features.loc[:,["experiment_id","control_bearing","timestamp","a1_x",
                                  "a1_y","a1_z","hz","w"]]
test_bearing = features.loc[:,["experiment_id","test_bearing","timestamp","a2_x",
                                  "a2_y","a2_z","hz","w"] ]

# Renaming columns on DF's

control_bearing = control_bearing.rename(columns={"control_bearing":"bearing_id","a1_x":"x_axis",
                                                  "a1_y":"y_axis","a1_z":"z_axis"})
test_bearing  = test_bearing.rename(columns={"test_bearing":"bearing_id","a2_x":"x_axis",
                                             "a2_y":"y_axis","a2_z":"z_axis"})

# Concatenating control and test group
df_features = pd.concat([control_bearing, test_bearing], axis=0)
print(df_features.shape)
print(target.shape)
print(df_features.columns)
print(target.columns)

# Merging "feature" DataFrame with "target" DataFrame
df = pd.merge(df_features, target)
df = df.rename(columns={"status":"target"})
print(df.shape)
print(df.isna().sum())