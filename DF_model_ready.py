import pandas as pd


# Allow wide tables
def initiate_pandas(max_cols, cons_width):
    pd.set_option('display.max_columns', max_cols)
    pd.set_option('display.max_rows', 250)
    pd.set_option('display.width', cons_width)  # make output in console wider

initiate_pandas(20, 1080)

features = pd.read_csv("bearing_signals.csv", sep=',')
target = pd.read_csv("bearing_classes.csv", sep=";")

# Renaming id-columns
# Dropping redundant "rpm"-column
features = features.rename(columns={"bearing_1_id": "control_bearing", "bearing_2_id": "test_bearing"})
# print(features.test_bearing.value_counts())


# Making two DF's: control and test group
control_bearing = features.loc[:, ["experiment_id", "control_bearing", "timestamp", "a1_x",
                                   "a1_y", "a1_z", "hz", "w", "rpm"]]
test_bearing = features.loc[:, ["experiment_id", "test_bearing", "timestamp", "a2_x",
                                "a2_y", "a2_z", "hz", "w", "rpm"]]

# Renaming columns on DF's

control_bearing = control_bearing.rename(columns={"control_bearing": "bearing_id", "a1_x": "x_axis",
                                                  "a1_y": "y_axis", "a1_z": "z_axis"})
test_bearing = test_bearing.rename(columns={"test_bearing": "bearing_id", "a2_x": "x_axis",
                                            "a2_y": "y_axis", "a2_z": "z_axis"})

# Concatenating control and test group
df_features = pd.concat([control_bearing, test_bearing], axis=0)

# Merging "feature" DataFrame with "target" DataFrame
df = pd.merge(df_features, target)
df = df.rename(columns={"status": "target"})


# get only rows up to 1.5 seconds using timeseries and store
df = df[df["timestamp"] < 1.5]

# Dropping rows with unusually high Hz and RPM
# In feature DataFrame
print(df.shape)
id_to_drop = [8,11,14,15,17,19,21,23,24,29,36,81]
index_to_drop = df[df["bearing_id"].isin(id_to_drop)].index
df = df.drop(index_to_drop)
print(df.shape)

# In target DataFrame
print(target.shape)
index_to_drop = target[target["bearing_id"].isin(id_to_drop)].index
target = target.drop(index_to_drop)
print(target.shape)

# Dropping irrelevant columns for model
print(df.shape)
df = df.drop(["experiment_id","rpm"], axis=1)
print(df.shape)
print(df.isna().sum())

print("""
Columns in feature.csv:""")
print(df.columns)

print("""
Columns in target.csv:""")
print(target.columns)
df.to_csv("csv_output/features.csv",index=False)
target.to_csv("csv_output/target.csv", index=False)