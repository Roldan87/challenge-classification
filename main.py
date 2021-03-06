from utils.manipulate_dataset import *
from utils.model import *
from scipy import fftpack

bear_class = pd.read_csv(r'..\data\bearing_classes.csv', sep=';')
bear_signal = pd.read_csv(r'..\data\bearing_signals.csv', sep=',')

if __name__ == '__main__':
    # manipulate dataset
    # low_speed_set = bear_signal[(bear_signal.timestamp > 0.25) &
    #                             (bear_signal.timestamp <= 1.5)]
    # low_speed_means = replace_axes_values_with_means(low_speed_set.copy())
    # low_speed_means = add_target_column(low_speed_means, bear_class)
    # write_datasets_to_csv({'vibration_test.csv': vibration_test, 'low_speed_set.csv': low_speed_set})

    # modeling RandomForest
    # features_to_exclude = ['target', 'experiment_id', 'bearing_1_id', 'bearing_2_id', 'hz', 'w']  # hz? rpm?
    # fit_evaluate_model_random_forest(low_speed_means, features_to_exclude, test_size=0.3)

    # feature engineering
    feature_engineering_df = extract_data_from_time_series_analysis(bear_signal,
                                                                    bear_class)
    sns.scatterplot(x=feature_engineering_df.mean_vibrations_x,
                    y=feature_engineering_df.max_amplitudes_x,
                    hue=feature_engineering_df.target)
    plt.show()
