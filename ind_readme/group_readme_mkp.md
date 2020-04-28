# Predicting Churn in Ride Share Company

Ben Weintraub, Eddie Ressegue, Maureen Petterson

## Intro
In an effort to retain ridership at a ride share company, we wanted to predict key factors affecting churn rate. Our dataset was pulled from July 1st, 2014 and contains data spanning the previous 5 months. The 12 features are:
- city
- sign-up date
- last trip date
- average distance
- average rating by driver
- average rating of driver
- surge percentage
- average surge
- trips in first 30 days
- luxury car user
- phone used for signup
- weekday percentage

Churn was defined as no activity within the past 30 days, eg, no rides during the month of June. 


## Exploratory Data Analysis and Data Preparation

<b>Data Preparation</b>

The dataset was alrady fairly clean, although there were 3 features with varying amounts of null values. 

<img alt="Data" src='img/data_head.png'>

Additionally, some of the features were categorical or contained information that was redundant. We made the following changes to both the train and test data:

- We filled in the missing values in 'average rating of driver' with the average rating from the other entries.  16% of the data in this column was missing and we felt this was too much data to drop from our analysis. 

- There were 201 nulls in the average rating of driver column. Similarly to the average rating by driver feature, we decided to fill in the nulls with the average value from other entries. 

- There were 396 nulls in the phone column, so we decided to drop those data points. 

- The city feature had three possible values (Winterfell, Astapor, and King's Landing) and we used one hot encoding to create three separate boolean features, one for each city. 

- We added a column for churn, with 1/True representing a customer who is not active in the past 30 days and 0 representing a customer who was active in the past 30 days. 

- We removed sign-up date, as these are all in January. We also removed last trip date as that data was no longer necessary. 


A screenshot of our cleaned dataset is below

<img alt="Clean Data" src='img/data_clean_head.png'>


<b>EDA</b>

Working on the training set only, we did some EDA to look at the distribution of the features. Below are a heatmap with correlation metrics, a histogram of numerical features, and a bar chart of the binary features. 


<img src="img/corr_heatmap.png" alt="HM" style="width: 600px;" align="center"/>

<img src="img/histograms_of_features.png" alt="HM" style="width: 800px;" align="center"/>



## Models Investigated

We chose 3 different models to test: Neural Networks, Random Forest, and Gradient Boosting Classification. We decided to evaluate our models on the following metrics: Accuracy, Precision, Recall, and the ROC curve. 


<b> Neural Networks</b>


<b> Random Forest</b>


<b> Gradient Boosting Classifier</b>

Out of the box metrics for Gradient Boosting Classfier were pretty good. The default values are:
- n_estimators = 100
- learning rate = 0.1
- max depth = 3

The results can be summarized in the confusion matrix below: 
<img src="img/confusion_matrix_gbc.png" alt="Drawing" style="width: 400px;" align="center"/>

<center>
    <b>Accuracy:</b> 79% | <b>Precision:</b> 86% | <b>Recall:</b> 81%
</center>
<br>
<br>


The Feature Importances are shown in the table below. 

<img src="img/feature_import_gbc.png" alt="Drawing" style="width: 400px;" align="center"/>
<br>
<br>

Optimizing Parameters: 

Looking at the training and testing errors as a function of number of trees leads to an optimized value of 830, although the change in test errors from 100 to 1000 is relatively minimal. The learning rate also affects the testing errors, but we found that the default learning rate of 0.1 actually works pretty well. 

<img alt="LR" src='img/errors_gbc.png'>


Running the "optimized" GBC model on our data results in the following ROC curve. 

<img src="img/roc_gbc.png" alt="Drawing" style="width: 400px;" align="center"/>
<br>

Summary of GBC: 

The Gradient Boosting Classifier works fairly well based on our scoring metrics; the area under the ROC curve is 0.85. We will need to compare this performance to that of the other models before selecting our optimal model for usage on the test data. 


The most influential features are: average rating by driver, surge percent, weekday percent, and living in King's Landing. Interestingly, these features were not highlighted in the correlation heatmap. 

## Comparison of Models











