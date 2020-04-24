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

Working on the training set only

<img alt="Heatmap" src='img/corr_heatmap.png'>
<img alt="Histograms" src='img/histograms_of_features.png'>




## Models Investigated

We choose 3 different models to test: Neural Networks, Random Forest, and Gradient Boosting Classification. We decided to evaluate our models on the following metrics: Accuracy, Precision, Recall, and the ROC curve. 


<b> Neural Networks</b>


<b> Random Forest</b>


<b> Gradient Boosting Classifier</b>




## Key Findings
 ( Identify interpret features that are the most influential in affecting
your predictions.)
( Consider business decisions that your model may indicate are appropriate.)





## Work Flow

1. Perform any cleaning, exploratory analysis, and/or visualizations to use the
provided data for this analysis.
   
2. Build a predictive model to help determine the probability that a rider will
be retained.

3. Evaluate the model.  Focus on metrics that are important for your *statistical
model*.
 


5. Discuss the validity of your model. Issues such as
leakage.  For more on leakage, see [this essay on
Kaggle](https://www.kaggle.com/dansbecker/data-leakage), and this paper: [Leakage in Data
Mining: Formulation, Detection, and Avoidance](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.7769&rep=rep1&type=pdf).

6. Repeat 2 - 5 until you have a satisfactory model.

7. Consider business decisions that your model may indicate are appropriate.
Evaluate possible decisions with metrics that are appropriate for *decision
rules*.
   