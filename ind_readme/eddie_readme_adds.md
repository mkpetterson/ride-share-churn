### Neural Network Model

<p align='middle'>
    <td><img src='../img/binary_bar_chart.png' align='center' style='width: 600px;'></td>
</p>


The idea behind the neural network model approach was to run a model with a large set of arbitrariliy engineered features. The original features were included in the model fit, as well as various operations on all non-binary feature types. These operations included:
- squaring
- cubing
- exponentiating
- taking the log
- inverting
- taking the sine and cosine
- multiplying all combinations of two features

This created a dataset with 110 total features. Because of this arbitrary feature engineering, feature importance was not investigated for this model.

Moderate tuning of the hyperparameters was done to achieve a final, satisfactory model. The input hyperparameters for the final neural network model include:
- 1000 epochs
- 3 hidden layers:
    - The first layer with ReLU activation
    - The second layer with TanH activation
    - The third layer with sigmoid activation to produce a probability
- 5000 batch size
- 0.2 validation split for cross validation
- 0.01 optimizer learning rate

<p align='middle'>
    <td><img src='../img/metrics_nn.png' align='center' style='width: 800px;'></td>
</p>

A confusion matrix with a threshold of 50% can be seen here:

<p align='middle'>
    <td><img src='../img/nn_conf_mat.png' align='center' style='width: 500px;'></td>
</p>

This matrix shows a large amount of false negatives, with a relatively small amount of false positives, so it is great at predicting positives but not negatives. This explains the high relative precision and low accuracy and recall.

<p align='middle'>
    <td><img src='../img/roc_nn.png' align='center' style='width: 500px;'></td>
</p>

The final model seems to make predictions in line with the Random Forest and Gradient Boosting models. However, the metrics here on the training data do not reflect similar values as the other models. The lack of interpretability of the neural network and the challenges in tuning the hyperparameters are among the reasons for not choosing to move forward with this model.