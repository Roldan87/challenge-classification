import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_spreads_per_axis(df: pd.DataFrame, df_classes: pd.DataFrame, axis: str) -> pd.DataFrame:
    # one index for each experiment
    a1_spreads = []
    a2_spreads = []
    a_differences = []

    # one loop per experiment
    for experiment_nr in np.arange(1, 113, step=1):
        experiment = df[df.experiment_id == experiment_nr]

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
    targets = df_classes.drop(index=0)['status'].reset_index(drop=True)

    data = {
        ('spread_a1_' + axis): series_a1_spreads,
        ('spread_a2_' + axis): series_a2_spreads,
        'difference_spread': series_a_differences,
        'target': targets}

    df_experiment_spreads = pd.DataFrame(data)
    return df_experiment_spreads



def save_aligned_swarmplots(df, df_classes, filename):
    x_spreads = get_spreads_per_axis(df, df_classes, 'x')
    y_spreads = get_spreads_per_axis(df, df_classes, 'y')
    z_spreads = get_spreads_per_axis(df, df_classes, 'z')

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
    savegif_path = 'visuals\\' + filename
    plt.savefig(savegif_path)
    plt.clf()


def save_stripplot_vibration_differences(axis: str,
                                         df: pd.DataFrame,
                                         df_classes: pd.DataFrame,
                                         experiment_nr: int) -> None:
    experiment = df[df.experiment_id == experiment_nr]
    status = df_classes[df_classes.index == experiment_nr].status

    if 1 in status.values:
        plt.title('control (green) versus test (blue) bearing (exp ' + str(experiment_nr) + ': good)', fontsize=14)
    else:
        plt.title('control (green) versus test (blue) bearing (exp ' + str(experiment_nr) + ': bad)', fontsize=14)

    column_name = 'a1_' + axis
    sns.stripplot(x=experiment.hz, y=experiment[column_name], color='green', alpha=0.35, jitter=0.3, dodge=True)
    column_name = 'a2_' + axis
    sns.stripplot(x=experiment.hz, y=experiment[column_name], color='blue', alpha=0.35, jitter=0.3, dodge=True)
    plt.ylim([-3, 3])
    plt.xlabel('rotations (hertz)')
    plt.ylabel('vibrations around ' + axis)
    savefig_name = 'strip_vibrations_' + axis + '_exp_nr_' + str(experiment_nr) + '.png'
    savegif_path = 'visuals\\' + savefig_name
    plt.savefig(savegif_path)
    plt.clf()


# save_aligned_swarmplots(low_speed_set, bear_class, 'swarm_lowspeed_vibrations.png')
# low_speed_for_plotting = low_speed_set[low_speed_set.hz < 4]
# for axis in ['x', 'y', 'z']:
#     save_stripplot_vibration_differences(axis, low_speed_set, bear_class, 2)
#     save_stripplot_vibration_differences(axis, low_speed_set, bear_class, 107)
