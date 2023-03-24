# Predicting Calories in a Recipe - Working Towards a Machine Learning Model


<br/>


## Problem Identification


In this project we will use the `Recipes and Ratings` dataset to create a Machine Learning model to do the following: Predict the `calories` in a recipe. Because `calories` is a continuous variable, this is a regression problem. 

In our investigation, `calories` is the response variable. We chose this as an extension of our work in Project 3 (linked at the top of this page), where we focused on the relationship between the `average rating` of a recipe and the `calories` of a recipe, but also briefly looked at the relationship between `calories` and the `number of steps` and the `number of ingredients` in a recipe. 

For our regression model, the metric we are using to evaluate how well it predicts `calories` is RMSE, although we will be including the R^2 values of our models in intermediate steps. The reason we chose to use RMSE instead of R^2 is that although R^2 gives us a good indication of the variance in `calories` that can be explained by our model, the RMSE value of our models when evaluated on the training and testing data will indicate how well the model predicts the actual values of `calories`.

The values in our dataset that are reasonable to include as features in our model, in that we would have access to this information about a recipe prior to/without knowing the calories of our recipe are:

- `minutes`: Estimated time to prepare the recipe.
- `nutrition`: Protein, fat, sugar, sodium, saturated fat, and carbohydrates in the recipe.
- `n_steps`: Number of steps in the recipe.
- `n_ingredients`: Number of ingredients in the recipe.
- `description`: String description of the recipe.
- `title`: String representing the title of the recipe.

We only needed to exclude data for the response variable, `calories`, which is the value in the first index of the `nutrition` list, since all other values can be known without knowing the `calories` of the recipe.


---
<br/>



## Baseline Model



Our baseline model predicts `calories` using the `number of ingredients` and the `number of steps` of a recipe. Both of these features are quantitative and discrete, and we did not conduct any encodings on these values.

We first create a test and train split on the dataset, with `test_size = 0.25`.


Towards creating the baseline:

In this step, we will create a basic model that predicts the `calories` in a recipe using only the  `number of ingredients` and the `number of steps` of the recipe.

In the training set, we first preprocessed the `number of ingredients` and the `number of steps` of a recipe using `StandardScaler()`, which standardized these values in each row relative to the values in the entire column for each feature. 

We then fit a linear regression model on the training data, with `X_train` being the  `number of ingredients` and the `number of steps` of the training set, and `y_train` being the corresponding `calories` of recipes in the training set. 

We then calculated the RMSE and R^2 values of this baseline model on the training set and the test set, which yielded the results seen below:

|      |   baseline train data |   baseline test data |
|:-----|----------------------:|---------------------:|
| RMSE |           174.573     |          174.619     |
| Rsq  |             0.0905381 |            0.0870858 |

<br/>

We believe that this baseline is not a great model to predict the `calories` of a recipe for various reasons. 

The R^2 values of this model for both the training set and the test set are less than 0.1, meaning that this model could only explain less than 10% of the variance in `calories` for both training and test data.

For our metric of interest, RMSE, the values for the training and test data are nearly identical, which is a good indicator that the model generalizes equally well to both training and test data, and is not overfitting to the training set.

However, the RMSE values of the baseline on both the training and test data is around 174 calories, while the standard deviation of `calories` for both is around 183 calories (the exact values can be seen below). 

|         |   Calories std |
|:--------|---------------:|
| y_train |        183.056 |
| y_test  |        182.758 |

<br/>

Because the RMSE values of this model are so close to the standard deviation of `calories`, for both the training and test data, it appears that the model is not too much better than a constant prediction of the mean of the `calories` column. 

Note: Recall that RMSE == Standard deviation if our only predictor is the mean.

---
<br/>


## Final Model



Our final model predicts `calories` using the following data:

- `number of ingredients` and the `number of steps` of a recipe, which are quantitative and discrete.
- `fat` and `protein` in a recipe in grams, which are quantitative and discrete. 
- the `title` and `description` strings of a recipe, from which we extracted specific words.

We continued to include the `number of ingredients` and the `number of steps` of a recipe in our model, since even in the linear regression model, these features improved the RMSE of the model, though slightly, as compared to simply predicting the mean, and did not overfit to the training model.

We decided to add the `fat` and `protein` of each recipe to the features used to make a prediction on `calories`, since these nutritional numbers are relevant to the `calories` of any recipe / food in general.

We then created three new feature columns in the training set as below:

- `fry`: contains `fry` if `fry` is in the steps of the recipe, else contains an empty string. We associated this word as likely being related to recipes with higher, and not lower, values of `calories`.
- `health`: contains `health` if `health` is in the title of the recipe, else contains an empty string. We associated this word as likely being related to recipes with lower, and not higher, values of `calories`.
- `low fat`: contains `low fat` if `low fat` is in the title of the recipe, else contains an empty string. We associated this word as likely being related to recipes with lower, and not higher, values of `calories`.

So overall, our final model uses the following feature columns: `number of ingredients`, `number of steps`, `fat`, `protein`, `fry`, `health`, `low fat`. 


We used the same test and train split of the dataset that we used to fit and evaluate the baseline model.


Towards creating the final model:

In this step, we will create a model that predicts the `calories` in a recipe using `number of ingredients`, `number of steps`, `fat`, `protein`, `fry`, `health`, `low fat`. 

Our final model of choice is `RandomForestRegressor`. We chose this model because we can easily tune the complexity of the model by adjusting the `max_depth` and `n_estimators` hyperparameters. We also made this choice because this model results in an average prediction over multiple decision trees, making it less likely to overfit as compared to a single decision tree, and also because it is less likely to overfit even when given a high number of features. 

In the training set, we first preprocessed the `number of ingredients` , `number of steps`, `protein`,and  `fat` of a recipe using `StandardScaler()`, which standardized these values in each row relative to the values in the entire column for each feature. 

We also preprocessed the `fry`, `health`, and `low fat` columns of each recipe using `OneHotEncoder`, and made sure to drop the columns that corresponded to One-Hot encodings of empty strings. 


After this preprocessing step, we wanted to find the best combination of the `max_depth` and `n_estimators` hyperparameters for our `RandomForestRegressor` given the selected features we are using to make a prediction. 


`max_depth` controls the maximum depth of each decision tree in the forest of our model, indicating the number of times the data is split into two groups until all leaf nodes contain similar data points. As max depth increases, the complexity of the model increases, and finding the optimal value is necessary to prevent under or over fitting. 


`n_estimators` is the number of decision trees in our random forest, where each tree is generated using a different resample of the training data, and the prediction made by the model is the aggregation of the results from the trees. As `n_estimators` increases, the complexity of the model increases, and finding the optimal value is necessary to prevent under or over fitting. 


We used a `GridSearchCV` to determine the best combination of `max_depth` and `n_estimators` hyperparameters, from `np.arange(2, 80, 20)` for `max_depth` and `np.arange(1, 15)` for `n_estimator`. The best parameters determined for our model by `GridSearchCV` were `max_depth = 22` and `n_estimators = 14`.

Predicting `calories` in the training set and the test set using this final fitted model resulted in RMSE and R^2 values as below. The final columns in this table show the difference between the RMSE and  R^2 values of the final and baseline model for the training set and the test set:

|      |   final train data |   final test data |
|:-----|-------------------:|------------------:|
| RMSE |         133.473    |        132.997    |
| Rsq  |           0.920818 |          0.693116 |


The performance of this final model is an improvement over the baseline, both in terms of our evaluation metric, the RMSE, as well as the Rsq  value on both the training and test set. 

|      |   final - base test |   final - base train |
|:-----|--------------------:|---------------------:|
| RMSE |          -41.6217   |           -41.0994   |
| Rsq  |            0.606031 |             0.830279 |


The Rsq  value of the final model on the training data is over 90% while it is almost 70% on the test data. This shows evidence of some overfitting on the training set, since the model explained more of the variance of `calories` in the training set than it did in the test set. However the Rsq  values of the final model on both the training and the test set are significantly better than those of the baseline, indicating that the final model, at the very minimum across both the training and test sets, explains 60% more of the variance in `calories` than the baseline.

For our evaluation metric, the RMSE of predictions of the final model for the training and test data are nearly identical, which is a good indicator that the model generalizes equally well to both training and test data, and is not overfitting to the training set. Additionally, the RMSE values of around 133 calories for the final model, are much less than the RMSE of approximately 174 calories for the baseline. Therefore, according to our evaluation metric, the final model does a better job than the baseline at predicting the values of the `calories` of a recipe, and therefore models the data better than the baseline.


---
<br/>


## Fairness Analysis




In this section, we will evaluate if our final model performs better in predicting `calories` for recipes that have a value of `fat` below the average value of `fat` in the dataset, as compared to predicting `calories` for recipes that have a value of `fat` above average.  

We will conduct all testing in this step on the test set of our data.

To the test set DataFrame, we assign a column called `fat_bracket` with different labels for if a recipe has `fat` content above or below average, a column containing the predictions of the final model for `calories` of that recipe, and a column containing the actual values of `calories` for that recipe.

As we have been doing through this analysis, our evaluation metric for how well our model predicts `calories` is RMSE. 

Grouping recipes based on whether or not they have `fat` content above the average, we see mean predictions of `calories` as below:

| fat_bracket   |   prediction |
|:--------------|-------------:|
| above_avg     |      441.173 |
| below_avg     |      205.782 |


Calculating the RMSEs of the model separately for recipes that have `fat` content above and below the average, we see values as below:

| fat_bracket   |    rmse |
|:--------------|--------:|
| above_avg     | 146.086 |
| below_avg     | 122.85  |


Our permutation test setup to check model performance on each of these groups is as below:

- Null Hypothesis: The RMSE of the final model is the same for both recipes with `fat` content above average and below average, and any    differences are due to chance.
- Alternative Hypothesis: The RMSE of the final model is lower for recipes with `fat` content below average.
- Test statistic: Difference in group RMSE 
    (RMSE(`fat` content above average) - RMSE(`fat` content below average)).
- Significance level: 0.01


The resulting p-value of this permutation test is `0.0`, and we thus reject the null hypothesis in favor of the alternative hypothesis, which states that our final model performed better in predicting `calories` for recipes with `fat` content below average.

The plot below visualizes the results of our permutation test.



<iframe src="grah.html" width=800 height=600 frameBorder=0></iframe>