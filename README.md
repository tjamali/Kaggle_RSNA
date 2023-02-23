Kaggle: RSNA Screening Mammography Breast Cancer Detection
=============

####  Competition Description

The objective of this contest is to detect breast cancer. The model should be trained on mammograms collected through routine screening. Efforts to improve the automation of detection in screening mammography may make radiologists more precise and productive, while enhancing the quality and safety of patient care. It might also aid in reducing medical expenses and unneeded treatments.

#### What we do

Here, I use the simple Linear Regression method to predict homes price. I will go through the following four steps:

1. Obtain the data
2. Explore the data and Feature engineering
3. Build a model
4. Make a submission file

### Step 1: Obtain the data

At the first step we need to download the competition [data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). 

#### File descriptions
- train.csv - the training set
- test.csv - the test set
- data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
- sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms

It is very useful to take a look at data_description.txt file that contains all information about the features in this competition.

The standard packages we use to make our enviroment are:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
We can use Pandas to read csv files as follows:

```py
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

The test dataframe contains 80 features, while the train dataframe has 81 features. The difference comes from this fact that the test data does not have the sale price column!

### Step 2: Explore the data and feature engineering

The main aim is to predict the sale price of the homes. In the training dataset, the SalePrice column (target variable) shows this information.

After analyzing data, we obtain useful information about the data. For instance, we will find that home price distribution is positively skewed. 

```py
plt.hist(train.SalePrice, bins = 20, color='blue')
plt.show()
```
![GitHub Logo](/images/hist1.png)

Since, we are going to use Linear Regression method and the data are skewed, it is better to take logarithm of the price values for removing skewness and getting more precise predictions.

Other important substeps are:

- Dealing with numeric features
 (We will find most correlated (positive/negative) features with sale prices and removing outliers)
- Dealing with null values
- Dealing with non-numeric features
- Transforming and engineering features
(We will use one-hot encoding to transform some non-numeric columns into a numeric ones)

**To see the details of each substeps please take a look at the Kaggle_House_SalePrice.ipynb.**

### Step 3: Build a linear model

At this step, we prepare the data for modeling by seperating the features (X) and the target variable (y) in the training dataset. We also split the original training dataset to the X_train, X_valid, y_train, y_valid. 

- X_train is the subset of features used for training.
- X_valid is the subset of features which will be used to test the model.
- y_train is the target variable SalePrice corresponding to X_train.
- y_valid is the target variable SalePrice corresponding to X_valid.

Partitioning the data into *train* and *valid* allows us to evaluate our model performance. This way, we can check that whether our model contains overfitting or not before applying it for the final prediction.

### Step 4: Make a submission file

At this point, we are ready to create a csv file including the predicted SalePrice for each observation in the test.csv dataset. After making submission file we should go to the [submission page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submit) to make a submission.

