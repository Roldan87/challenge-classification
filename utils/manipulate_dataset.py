import pandas as pd
import numpy as np
from scipy import fftpack

def separate_limit_test_experiments_from_vibration_test_experiments(df, experiments_list):
    df_copy = df.copy()
    for experiment_number in experiments_list:
        df_copy.drop(df_copy[df_copy['experiment_id'] == experiment_number].index, inplace=True)
    return df_copy


def drop_hertz_outliers(df, df_classes):
    df_copy = df.copy()
    hertz_outlier_experiments = df[df.hz > 100]
    hertz_outlier_experiments = hertz_outlier_experiments.groupby('experiment_id', as_index=False).agg('mean')
    df_limit_test_features = df[df.experiment_id.isin(hertz_outlier_experiments.experiment_id.values)]
    df_signals_dropped_outliers = df.drop(
        df[df.experiment_id.isin(df_limit_test_features.experiment_id.unique())].index)
    df_limit_test_classes = df[df.index.isin(hertz_outlier_experiments.experiment_id.values)]
    df_classes_dropped_outliers = df_classes.drop(
        df_classes[df_classes.index.isin(df_limit_test_features.experiment_id.unique())].index)

    return df_signals_dropped_outliers, df_classes_dropped_outliers, df_limit_test_classes


def write_datasets_to_csv(*dataframe_dictionary):
    for dictionary in dataframe_dictionary:
        for key in dictionary.keys():
            df = dictionary[key]
            df.to_csv(key)


def replace_axes_values_with_means(df):
    low_speed_means = pd.DataFrame()
    for experiment_id in df.experiment_id.unique():
        df_experiment = df[df.experiment_id == experiment_id]

        mean_a1_x = df_experiment.a1_x.mean()
        df_experiment['a1_x'] = mean_a1_x

        mean_a1_y = df_experiment.a1_y.mean()
        df_experiment['a1_y'] = mean_a1_y

        mean_a1_z = df_experiment.a1_z.mean()
        df_experiment['a1_z'] = mean_a1_z

        mean_a2_x = df_experiment.a2_x.mean()
        df_experiment['a2_x'] = mean_a2_x

        mean_a2_y = df_experiment.a2_y.mean()
        df_experiment['a2_y'] = mean_a2_y

        mean_a2_z = df_experiment.a2_z.mean()
        df_experiment['a2_z'] = mean_a2_z

        low_speed_means = pd.concat([low_speed_means, df_experiment], axis=0)

    return low_speed_means


def add_target_column(df, df_classes):
    df_merged = df.merge(df_classes, left_on='experiment_id', right_on=df_classes.index)
    df_merged.rename(columns={'status':'target'}, inplace=True)
    return df_merged.drop('bearing_id', axis=1)

def fft_dataframe_maker(df):
    exp_id = 0
    c_exp = df[df['bearing_2_id'] == 1]
    c_sig_x = c_exp.a1_x.values
    c_sig_y = c_exp.a1_y.values
    c_sig_z = c_exp.a1_z.values
    c_sig_fft_x = np.abs(fftpack.fft(c_sig_x))
    c_sig_fft_y = np.abs(fftpack.fft(c_sig_y))
    c_sig_fft_z = np.abs(fftpack.fft(c_sig_z))
    control_freq_list = {'Exp_id': exp_id, 'fft_X': c_sig_fft_x, 'fft_Y': c_sig_fft_y, 'fft_Z': c_sig_fft_z}

    df_freq = pd.DataFrame(control_freq_list)

    for i in range(2,113):
        exp = df[df['bearing_2_id'] == i]
        sig_x = exp.a2_x.values
        sig_y = exp.a2_y.values
        sig_z = exp.a2_z.values
        sig_fft_x = np.abs(fftpack.fft(sig_x))
        sig_fft_y = np.abs(fftpack.fft(sig_y))
        sig_fft_z = np.abs(fftpack.fft(sig_z))
        exp_freq_list = {'Exp_id': i, 'fft_X': sig_fft_x, 'fft_Y': sig_fft_y, 'fft_Z': sig_fft_z}
        df_temp = pd.DataFrame(exp_freq_list)
        df_freq = pd.concat([df_freq, df_temp], axis=0)