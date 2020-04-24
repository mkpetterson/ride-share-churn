<p align='middle'>
    <td><img src='img/binary_bar_chart.png' align='center' style='width: 600px;'></td>
</p>


The idea behind the neural network model approach was to run a model with a large set of arbitrariliy engineered features. The original features were included in the model fit, as well as various operations on all non-binary feature types. These operations included:
- squaring
- cubing
- exponentiating
- taking the sine and cosine
- multiplying all combinations of two features

This creted a dataset with 97 total features. Because of this arbitrary feature engineering, feature importance was not investigated for this model.

The input hyperparameters for the final neural network model include:
- 1000 epochs
- 3 hidden layers:
    - The first layer with ReLU activation
    - The second layer with TanH activation
    - The third layer with sigmoid activation to produce a probability
- 5000 batch size
- 0.2 validation split for cross validation
- 0.01 optimizer learning rate

<p align='middle'>
    <td><img src='img/metrics_nn.png' align='center' style='width: 800px;'></td>
</p>

Moderate tuning of the hyperparameters was done to achieve this final model.


<p align='middle'>
    <td><img src='img/roc_nn.png' align='center' style='width: 500px;'></td>
</p>

The ROC curve resides in the same range as the other models analyzed.