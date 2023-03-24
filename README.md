# Predicting Calories in a Recipe - Working Towards a Machine Learning Model


<br/>


## Problem Identification


In this project we will use the `Recipes and Ratings` dataset to create a Machine Learning model to do the following: Predict the `calories` in a recipe. Because `calories` is a continuous variable, this is a regression problem. 

In our investigation, `calories` is the response variable. We chose this as an extension of our work in Project 3 (linked at the top of this page), where we focused on the relationship between the `average rating` of a recipe and the `calories` of a recipe, but also briefly looked at the relationship between `calories` and the `number of steps` and the `number of ingredients` in a recipe. 

For our regression model, the metric we are using to evaluate how well it predicts `calories` is RMSE, although we will be including the R<sup>2</sup> values of our models in intermediate steps. The reason we chose to use RMSE instead of R<sup>2</sup> is that although R<sup>2</sup> gives us a good indication of the variance in `calories` that can be explained by our model, the RMSE value of our models when evaluated on the training and testing data will indicate how well the model predicts the actual values of `calories`.

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


<br/>


Our baseline model predicts `calories` using the `number of ingredients` and the `number of steps` of a recipe. Both of these features are quantitative and discrete, and we did not conduct any encodings on these values.

We first create a test and train split on the dataset, with `test_size = 0.25`.


Towards creating the baseline:

In this step, we will create a basic model that predicts the `calories` in a recipe using only the  `number of ingredients` and the `number of steps` of the recipe.

In the training set, we first preprocessed the `number of ingredients` and the `number of steps` of a recipe using `StandardScaler()`, which standardized these values in each row relative to the values in the entire column for each feature. 

We then fit a linear regression model on the training data, with `X_train` being the  `number of ingredients` and the `number of steps` of the training set, and `y_train` being the corresponding `calories` of recipes in the training set. 

We then calculated the RMSE and R<sup>2</sup> values of this baseline model on the training set and the test set, which yielded the results seen below:

|   baseline train data |   baseline test data |
|----------------------:|---------------------:|
|           174.402     |          175.125     |
|             0.0890051 |            0.0917294 |

<br/>

We believe that this baseline is not a great model to predict the `calories` of a recipe for various reasons. 

The R<sup>2</sup> values of this model for both the training set and the test set are less than 0.1, meaning that this model could only explain less than 10% of the variance in `calories` for both training and test data.

For our metric of interest, RMSE, the values for the training and test data are nearly identical, which is a good indicator that the model generalizes equally well to both training and test data, and is not overfitting to the training set.

However, the RMSE values of the baseline on both the training and test data is around 174 calories, while the standard deviation of `calories` for both is around 183 calories (the exact values can be seen below). 

|   Calories std |
|---------------:|
|        182.723 |
|        183.755 |

<br/>

Because the RMSE values of this model are so close to the standard deviation of `calories`, for both the training and test data, it appears that the model is not too much better than a constant prediction of the mean of the `calories` column. 

Note: Recall that RMSE == Standard deviation if our only predictor is the mean.

---
<br/>


## Final Model


<br/>


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






