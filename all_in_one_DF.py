import pandas as pd


# Allow wide tables
def initiate_pandas(max_cols, cons_width):
    pd.set_option('display.max_columns', max_cols)
    pd.set_option('display.max_rows', 250)
    pd.set_option('display.width', cons_width)  # make output in console wider


initiate_pandas(20, 1080)

# Reading ONLY 100k rows.
features = pd.read_csv("bearing_signals.csv", sep=',', nrows=100000)
target = pd.read_csv("bearing_classes.csv", sep=";")

# Cleaning datatypes
target["status"] = target["status"].astype(bool)
# convert floats to recognized time format
features["timestamp"] = pd.to_timedelta(features.timestamp, unit="s")

# Renaming id-columns
# Dropping redundant "rpm"-column
features = features.rename(columns={"bearing_1_id": "control_bearing", "bearing_2_id": "test_bearing"})
features = features.drop(columns=["rpm"], axis=1)
print(features.columns)
# print(features.test_bearing.value_counts())

# Making two DF's: control and test group
control_bearing = features.loc[:, ["experiment_id", "control_bearing", "timestamp", "a1_x",
                                   "a1_y", "a1_z", "hz", "w"]]
test_bearing = features.loc[:, ["experiment_id", "test_bearing", "timestamp", "a2_x",
                                "a2_y", "a2_z", "hz", "w"]]

# Renaming columns on DF's

control_bearing = control_bearing.rename(columns={"control_bearing": "bearing_id", "a1_x": "x_axis",
                                                  "a1_y": "y_axis", "a1_z": "z_axis"})
test_bearing = test_bearing.rename(columns={"test_bearing": "bearing_id", "a2_x": "x_axis",
                                            "a2_y": "y_axis", "a2_z": "z_axis"})

# Concatenating control and test group
df_features = pd.concat([control_bearing, test_bearing], axis=0)
print(df_features.shape)
print(target.shape)
print(df_features.columns)
print(target.columns)

# narrowing features to first 1.5 seconds
features = features[features["timestamp"].dt.total_seconds() < 1.5]

# Merging "feature" DataFrame with "target" DataFrame
# df = pd.merge(df_features, target)
df = pd.merge(left=target, right=features, left_on="bearing_id", right_on="control_bearing")
df = df.rename(columns={"status": "target"})
print(df.shape)
print(df.isna().sum())

# get only rows up to 1.5 seconds using timeseries and store
df.to_csv("csv_output/focus.csv")
print(features.shape)
