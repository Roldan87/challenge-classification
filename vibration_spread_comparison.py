import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


bearing_signals = pd.read_csv(r'..\archive\bearing_signals.csv')
bearing_classes = pd.read_csv(r'..\archive\bearing_classes.csv', sep=';')


def split_dataframes_to_drop_hertz_outliers(df, df_classes):
    hertz_outlier = df[df.hz > 100]
    hertz_outlier = hertz_outlier.groupby('experiment_id', as_index=False).agg('mean')
    df_limit_test_features = df[df.experiment_id.isin(hertz_outlier.experiment_id.values)]
    df_signals_dropped_outliers = df.drop(
        df[df.experiment_id.isin(df_limit_test_features.experiment_id.unique())].index)
    df_limit_test_classes = df[df.index.isin(hertz_outlier.experiment_id.values)]
    df_classes_dropped_outliers = df_classes.drop(
    df_classes[df_classes.index.isin(df_limit_test_features.experiment_id.unique())].index)

    return df_signals_dropped_outliers, df_classes_dropped_outliers, df_limit_test_classes


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
    plt.clf()  # clear buffer of matplotlib as to not overlap plots
    sns.boxplot(x='target', y='difference_spread', data=spreads_df)
    plt.ylim([-5, 30])
    plt.title('Vibration spreads on axis ' + axis)
    plt.xlabel('target')
    plt.ylabel('differences between max_spread of a2 and a1')
    savefig_name = 'vibration_spreads_' + axis + '.png'
    savegif_path = 'visuals\\' + savefig_name
    plt.savefig(savegif_path)
    plt.clf()


def save_aligned_box_plots(x_spreads, y_spreads, z_spreads):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8), sharey='all')
    plt.ylim([-10, 30])
    fig.suptitle('Vibration spread differencess on axes x, y, z', fontsize=17)
    fig.text(0.5, 0.02, 'bad (0) or good (1) bearing', fontsize=13, ha='center')
    plt.ylabel('differences between max_spread of a2 and a1')
    fig.text(0.07, 0.5,
         'difference between max_min_spread of test and control bearing',
         fontsize=13,
         rotation='vertical',
         va='center')
    x = sns.boxplot(ax=ax1, x='target', y='difference_spread', data=x_spreads, palette='Set2')
    x.set(xlabel='x', ylabel=None)
    y = sns.boxplot(ax=ax2, x='target', y='difference_spread', data=y_spreads, palette='Set2')
    y.set(xlabel='y', ylabel=None)
    z = sns.boxplot(ax=ax3, x='target', y='difference_spread', data=z_spreads, palette='Set2')
    z.set(xlabel='z', ylabel=None)
    savefig_name = 'vibration_spread_differences_on_all_axes.png'
    savegif_path = 'visuals\\' + savefig_name
    plt.savefig(savegif_path)
    plt.clf()


def save_aligned_swarmplots(x_spreads, y_spreads, z_spreads):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8), sharey='all')
    plt.ylim([-10, 30])
    fig.suptitle('Vibration spread differencess on axes x, y, z', fontsize=17)
    fig.text(0.5, 0.02, 'bad (0) or good (1) bearing', fontsize=13, ha='center')
    plt.ylabel('differences between max_spread of a2 and a1')
    fig.text(0.07, 0.5,
         'difference between max_min_spread of test and control bearing',
         fontsize=13,
         rotation='vertical',
         va='center')
    x = sns.swarmplot(ax=ax1, x='target', y='difference_spread', data=x_spreads, palette='Set2', dodge=True)
    x.set(xlabel='x', ylabel=None)
    y = sns.swarmplot(ax=ax2, x='target', y='difference_spread', data=y_spreads, palette='Set2', dodge=True)
    y.set(xlabel='y', ylabel=None)
    z = sns.swarmplot(ax=ax3, x='target', y='difference_spread', data=z_spreads, palette='Set2', dodge=True)
    z.set(xlabel='z', ylabel=None)
    savefig_name = 'swarmplot_vibration_spread_differences_on_all_axes.png'
    savegif_path = 'visuals\\' + savefig_name
    plt.savefig(savegif_path)
    plt.clf()


def sum_values_for_all_axes(experiment_nr, df):
    df_experiment = df[df.experiment_id == experiment_nr]

    for axis in ['x', 'y', 'z']:
        sum_vibration_axis_control = []
        previous_abs_a1 = 0
        sum_vibration_axis_test = []
        previous_abs_a2 = 0

        for idx, row in df_experiment.iterrows():
            column_name = 'a1_' + axis
            abs_a1 = abs(row[column_name])
            summed_value_a1 = previous_abs_a1 + abs_a1
            sum_vibration_axis_control.append(summed_value_a1)
            previous_abs_a1 = summed_value_a1

            column_name = 'a2_' + axis
            abs_a2 = abs(row[column_name])
            summed_value_a2 = previous_abs_a2 + abs_a2
            sum_vibration_axis_test.append(summed_value_a2)
            previous_abs_a2 = summed_value_a2

        column_name = 'sum_a1_' + axis
        df_experiment[column_name] = sum_vibration_axis_control

        column_name = 'sum_a2_' + axis
        df_experiment[column_name] = sum_vibration_axis_test
    return df_experiment


def get_df_of_first_seconds(df):
    # isolate first 1.5 seconds with low power
    condition_seconds = df.timestamp < 5
    condition_watts = df.hz < 4
    df_before_powering_up = df[(condition_seconds) & (condition_watts)]
    return df_before_powering_up


def save_plot_differences_vibration_beginnings(axis: str,
                                                    df: pd.DataFrame,
                                                    df_classes: pd.DataFrame,
                                                    experiment_nr: int) -> None:

    experiment = df[df.experiment_id == experiment_nr]
    status = df_classes[df_classes.index == experiment_nr].status
    if 1 in status.values:
        plt.title('good test bearing')
    else:
        plt.title('bad test bearing')

    column_name = 'a1_' + axis
    plt.plot(experiment.timestamp, experiment[column_name], color='green', alpha=0.8, label=('control ' + axis))
    column_name = 'a2_' + axis
    plt.plot(experiment.timestamp, experiment[column_name], color='blue', alpha=0.8, label=('test ' + axis))
    plt.plot(experiment.timestamp, experiment.hz, color='yellow', alpha=0.5, label='hz')
    plt.legend()
    savefig_name = 'beginning_vibrations_' + axis + '_exp_nr_' + str(experiment_nr) + '.png'
    savegif_path = 'visuals\\' + savefig_name
    plt.savefig(savegif_path)
    plt.clf()


def save_swarmplot_differences_vibration_beginnings(axis: str,
                                                    df: pd.DataFrame,
                                                    df_classes: pd.DataFrame,
                                                    experiment_nr: int) -> None:

    experiment = df[df.experiment_id == experiment_nr]
    status = df_classes[df_classes.index == experiment_nr].status

    if 1 in status.values:
        plt.title('good test bearing')
    else:
        plt.title('bad test bearing')

    column_name = 'a1_' + axis
    sns.swarmplot(x=experiment.hz, y=experiment[column_name], color='green', alpha=0.5, dodge=True)
    column_name = 'a2_' + axis
    sns.swarmplot(x=experiment.hz, y=experiment[column_name], color='blue', alpha=0.5, dodge=True)
    plt.legend()
    plt.ylim([-3, 3])
    savefig_name = 'swarm_beginning_vibrations_' + axis + '_exp_nr_' + str(experiment_nr) + '.png'
    savegif_path = 'visuals\\' + savefig_name
    plt.savefig(savegif_path)
    plt.clf()


###################################################################
###################################################################

bearings_signals_dropped_outliers, bearing_classes_dropped_outliers, dropped_bearings = \
    split_dataframes_to_drop_hertz_outliers(bearing_signals, bearing_classes)

# add columns with vibrations sums for all axes for plotting over the whole experiment (not just the beginning seconds)
good_bearing_101 = sum_values_for_all_axes(101, bearings_signals_dropped_outliers)
bad_bearing_98 = sum_values_for_all_axes(98, bearings_signals_dropped_outliers)

bearings_signals_experiments_beginnings = get_df_of_first_seconds(bearings_signals_dropped_outliers)


# save_plot_differences_vibration_beginnings('x',
#                                            bearings_signals_experiments_beginnings,
#                                            bearing_classes_dropped_outliers,
#                                            98)    # bad bearing
# save_plot_differences_vibration_beginnings('x',
#                                            bearings_signals_experiments_beginnings,
#                                            bearing_classes_dropped_outliers,
#                                            101)    # good bearing
# save_swarmplot_differences_vibration_beginnings('x',
#                                                 bearings_signals_experiments_beginnings,
#                                                 bearing_classes_dropped_outliers,
#                                                 98)    # bad
# save_swarmplot_differences_vibration_beginnings('x',
#                                                 bearings_signals_experiments_beginnings,
#                                                 bearing_classes_dropped_outliers,
#                                                 101)    # good


# x_spreads = get_spreads_per_axis('x')
# write_to_csv(x_spreads, 'x')
# save_boxplot(x_spreads, 'x')

# y_spreads = get_spreads_per_axis('y')
# write_to_csv(y_spreads, 'y')
# save_boxplot(y_spreads, 'y')

# z_spreads = get_spreads_per_axis('z')
# write_to_csv(z_spreads, 'z')
# save_boxplot(z_spreads, 'z')

# save_aligned_box_plots(x_spreads, y_spreads, z_spreads)
# save_aligned_swarmplots(x_spreads, y_spreads, z_spreads)
