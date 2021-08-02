## Challenge Bearing Classification

# Description
  This was an assignment we received during our training at BeCode.  
  The main goal was to get used to Machine Learning, specifically using classification algorithms.   
  For this we used a database from <a href="https://www.kaggle.com/isaienkov/bearing-classification" target="_blank">Kaggle</a>, on testing bearings.  
  Our job was to predict if a bearing was defective or not, with an accuracy as high as possible.
  
  
# Installation
## Python version
* Python 3.9

## Databases
* <a href="https://www.kaggle.com/isaienkov/bearing-classification?select=bearing_classes.csv" target="_blank">Target</a>
* <a href="https://www.kaggle.com/isaienkov/bearing-classification?select=bearing_signals.csv" target="_blank">Features</a>

## Packages used
* pandas
* numpy
* matplotlib.pyplot
* seaborn
* sklearn

# Usage
| File     | Description                                                   |
|----------|---------------------------------------------------------------|
| main.py  | File containing Python code.    <br>Used for cleaning and feature engineering the data |
| plots.py | File containing Python code.   <br>Used for getting to know the data and finding any   <br>correlations between features |
| model.py | File containing Python code, using ML - Random Forest.   <br>Fitting our data to the model and use to it make predictions. |

# Feature engineering
| Column name of feature | Change made                    | Reason                                                                                                                        |
|------------------------|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| timestamp              | Only keeping rows above 0,1620 | We found some outliers where the "rpm" and "hz" values spiked in the first parts of the test.  <br>With the use of plotting, we discovered a cut off point. |
| timestamp              | Only keeping rows above 1,5    | We found that the biggest differences between it being a bad or good bearing,  could be found in the first parts of the test.  <br>With the use of plotting, we discovered a cut off point. |





# Visuals
## Machine used to gather the data on bearings

![](visuals/bearing_test_machine.jpg)

## Plot showing the min-max-difference of every axis, on every bearing.

![](visuals/vibration_spread_differences_on_all_axes.png)


# Contributors
| Name                  | Github                                 |
|-----------------------|----------------------------------------|
| Patrick Brunswyck        | https://github.com/brunswyck               |
| Jose Roldan | https://github.com/Roldan87 |
| Matthew Samyn         | https://github.com/matthew-samyn       |
| Maarten Van den Bulcke           | https://github.com/MaartenVdBulcke       |




# Timeline
29/07/2021 - 03/08/2021
