import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

bearing_signals = pd.read_csv(r'..\archive\bearing_signals.csv')
bearing_classes = pd.read_csv(r'..\archive\bearing_classes.csv', sep=';')


def get_spreads_per_axis(axis: str) -> pd.DataFrame:
    # one index for each experiment
    a1_spreads = []
    a2_spreads = []
    a_differences = []

    # one loop per experiment
    for experiment_nr in np.arange(1, 113, step=1):
        experiment = bearing_signals[bearing_signals.experiment_id == experiment_nr]

        # search max, min, difference for a1
        columns_name = 'a1_' + axis
        a1_max = experiment[columns_name].max()
        a1_min = experiment[columns_name].min()
        min_max_spread_a1 = a1_max - a1_min

        # search max, min, difference for a2
        columns_name = 'a2_' + axis
        a2_max = experiment[columns_name].max()
        a2_min = experiment[columns_name].min()
        min_max_spread_a2 = a2_max - a2_min

        # add all to lists
        a1_spreads.append(min_max_spread_a1)
        a2_spreads.append(min_max_spread_a2)
        a_differences.append(min_max_spread_a2 - min_max_spread_a1)

    series_a1_spreads = pd.Series(a1_spreads)
    series_a2_spreads = pd.Series(a2_spreads)
    series_a_differences = pd.Series(a_differences)

    # drop first row of targets for bearing_id 0
    targets = bearing_classes.drop(index=0)['status'].reset_index(drop=True)

    data = {
        ('spread_a1_' + axis): series_a1_spreads,
        ('spread_a2_' + axis): series_a2_spreads,
        'difference_spread': series_a_differences,
        'target': targets}

    df_experiment_spreads = pd.DataFrame(data)
    return df_experiment_spreads


def write_to_csv(spreads_df, axis):
    csv_name = 'spreads_a1a2_' + axis + '.csv'
    csv_path = 'csv_output\\' + csv_name
    spreads_df.to_csv(csv_path)


def save_boxplot(spreads_df: pd.DataFrame, axis: str):
    sns.boxplot(x='target', y='difference_spread', data=spreads_df)
    plt.ylim([-5, 30])
    plt.title('Vibration spreads on axis ' + axis)
    plt.xlabel('target')
    plt.ylabel('differences between max_spread of a2 and a1')
    savefig_name = 'vibration_spreads_' + axis + '.png'
    savegif_path = 'visuals\\' + savefig_name
    plt.savefig(savegif_path)


x_spreads = get_spreads_per_axis('x')
write_to_csv(x_spreads, 'x')
save_boxplot(x_spreads, 'x')

y_spreads = get_spreads_per_axis('y')
write_to_csv(y_spreads, 'y')
save_boxplot(y_spreads, 'y')

z_spreads = get_spreads_per_axis('z')
write_to_csv(z_spreads, 'z')
save_boxplot(z_spreads, 'z')
