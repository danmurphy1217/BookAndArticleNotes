# Introduction to Regression
Here, the target value is a continuously varying variable like the price of a house or a country's GDP. <br/>


### Load the data and reshape it:
`# Import numpy and pandas`<br/>
`import numpy as np`<br/>
`import pandas as pd`<br/>

`# Read the CSV file into a DataFrame: df`<br/>
`df = pd.read_csv('gapminder.csv')`<br/>

`# Create arrays for features and target variable`<br/>
`y = df['life'].values`<br/>
`X = df['fertility'].values`<br/>

`# Print the dimensions of X and y before reshaping`<br/>
`print("Dimensions of y before reshaping: {}".format(y.shape))`<br/>
`print("Dimensions of X before reshaping: {}".format(X.shape))`<br/>

`# Reshape X and y`<br/>
`y = y.reshape(-1, 1)`<br/>
`X = X.reshape(-1, 1)`<br/>

`# Print the dimensions of X and y after reshaping`<br/>
`print("Dimensions of y after reshaping: {}".format(y.shape))`<br/>
`print("Dimensions of X after reshaping: {}".format(X.shape))` <b r/>

# Correlation between features
The Seaborn heatmap function can be very useful for these things!
`df.corr()` will calculate the pairwise correlation between columns <br/>
`sns.heatmap(df.corr(), square=True, cmap='RdYlGn')`<br/>
**In linear regression (not multi-linear reg.), the question is how do we choose the a and b for our y = ax + b model. What a and b are we testing?**
Then, we must choose the cost function (or error function). We want to calculate the distance between each data point and our estimated line (OLS or Mean of Squared Errors). Calling .fit() in SKL performs OLS under the hood for us. 
**In higher dimensions, we have y = a1X1 + a2X2 + ... + b**
**use .score() to test the performance of the lin reg model**


# Fit & Predict for regression:
`# Import LinearRegression`<br>
`from sklearn.linear_model import LinearRegression`<br/>

`# Create the regressor: reg`<br/>
`reg = LinearRegression()`<br/>

`# Create the prediction space`<br/>
`prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)` <br/>

`# Fit the model to the data`<br/>
`reg.fit(X_fertility, y)`<br/>

`# Compute predictions over the prediction space: y_pred`<br/>
`y_pred = reg.predict(prediction_space)`<br/>

`# Print R^2 `<br/>
`print(reg.score(X_fertility, y))`<br/>

`# Plot regression line` <br/>
`plt.plot(prediction_space, y_pred, color='black', linewidth=3)`<br/>
`plt.show()`<br/>


# Train/Test/Split for Regression
`# Import necessary modules`<br/>
`from sklearn.linear_model import LinearRegression`<br/>
`from sklearn.metrics import mean_squared_error`<br/>
`from sklearn.model_selection import train_test_split`<br/>

`# Create training and test sets`<br/>
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)`<br/>

`# Create the regressor: reg_all`<br/>
`reg_all = LinearRegression()`<br/>

`# Fit the regressor to the training data`<br/>
`reg_all.fit(X_train, y_train)`<br/>

`# Predict on the test data: y_pred`<br/>
`y_pred = reg_all.predict(X_test)`<br/>

`# Compute and print R^2 and RMSE`<br/>
`print("R^2: {}".format(reg_all.score(X_test, y_test)))`<br/>
`rmse = np.sqrt(mean_squared_error(y_test, y_pred))`<br/>
`print("Root Mean Squared Error: {}".format(rmse))`<br/>

# Cross-Validation:
Cross-Validation is used to combat the arbitrary split in the data that we create when we use train_test_split. Specifically, the results we get with R^2 are dependent on the way we split the data, and it may not be representative of how our model acts on testing data. 

## Steps:
1. Create 5 groups (folds)
2. Hold out first fold as test set, fit our model on the other four folds, predict on the first fold, and compute the metric of interest. Then, we hold out the second fold, use the other four folds to fit the model, and then predict using the second fold. We repeat this for each fold. Then, we have 5 R^2 values that we can use to generalize across our data, create confidence intervals, and more. 
3. Using 5 folds is called 5-fold CV, 10 folds is 10-fold CV, and so on. But, if we use K folds, its called K-fold CV. More folds is more computationally expensive because we are fitting and predicting more times.
`from sklearn.model_selection import cross_val_score`<br/>
`from sklearn.linear_model import LinearRegression`<br/>
`reg = LinearRegression()`<br/>
`cv_results = cross_val_scores(reg, X, y, cv=5)` <br/>
Here, **cv** is the number of folds we want to create. cv_results will, when cv= 5, report 5 R^2 values for linear regression. Then, we can create the mean from those values using: <br/>
`np.mean(cv_results)`<br/>

# EXAMPLE CODE:
`# Import the necessary modules`<br/>
`from sklearn.model_selection import cross_val_score`<br/>
`from sklearn.linear_model import LinearRegression`<br/>

`# Create a linear regression object: reg`<br/>
`reg = LinearRegression()`<br/>

`# Compute 5-fold cross-validation scores: cv_scores`<br/>
`cv_scores = cross_val_score(reg, X, y, cv = 5)`<br/>

`# Print the 5-fold cross-validation scores`<br/>
`print(cv_scores)`<br/>

`print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))`<br/>

# 3-fold vs. 10-fold cross val score:
`from sklearn.linear_model import LinearRegression`<br/>
`from sklearn.model_selection import cross_val_score`<br/>

`# Create a linear regression object: reg`<br/>
`reg = LinearRegression()`<br/>

`# Perform 3-fold CV`<br/>
`cvscores_3 = cross_val_score(reg, X, y, cv =3)`<br/>
`print(np.mean(cvscores_3))`<br/>

`# Perform 10-fold CV`<br/>
`cvscores_10 = cross_val_score(reg, X, y, cv= 10)`<br/>
`print(np.mean(cvscores_10))`<br/>

# Regularized Reg
Linear reg minizes a loss function. It chooses a coefficient for each variable. Large coefficients can be penalized with regularized regression. Ridge regression  uses the OLS function plus the sum of each coefficient squared times some alpha (that we choose, it can also be called lambda). Use normalize to ensure all our variables are on the same scale. <br/>
Lasso regression can get rid of coefficients/features that are not important! To visualize the important features: <br />
`from sklearn.linear_model import Lasso`<br/>
`names = dataset.drop('COLNAME', axis=1).columns`<br/>
`lasso = Lasso(alpha=0.1)`<br/>
`lasso_coef = lasso.fit(X, y).coef_`<br/>
`_ = plt.plot(range(len(names)), lasso_coef)`<br/>
`_ = plt.xticks(range(len(names)), names, rotation = 60)`<br/>
`_ = plt.ylabel('Coefficients)` <br/>
`plt.show()`<br/>

This is very useful as it can show the most important feature!!! We must know how to do this for datasets. 

# Visualize the most important feature(s):
`# Import Lasso` <br/>
`from sklearn.linear_model import Lasso`<br/>
`# Instantiate a lasso regressor: lasso`<br/>
`lasso = Lasso(alpha = 0.4, normalize=True)`<br/>

`# Fit the regressor to the data`<br/>
`lasso.fit(X, y)`<br/>

`# Compute and print the coefficients`<br/>
`lasso_coef = lasso.fit(X, y).coef_ # fit the regressor and calculate the coefficients (using .coef_)`<br/>
`print(lasso_coef)`<br/>

`# Plot the coefficients`<br/>
`plt.plot(range(len(df_columns)), lasso_coef)`<br/>
`plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)`<br/>
`plt.margins(0.02)`<br/>
`plt.show()`<br/>

# Ridge Regression:
`# Import necessary modules`<br/>
`from sklearn.linear_model import Ridge`<br/>
`from sklearn.model_selection import cross_val_score`<br/>

`# Setup the array of alphas and lists to store scores`<br/>
`alpha_space = np.logspace(-4, 0, 50)`<br/>
`ridge_scores = []`<br/>
`ridge_scores_std = []`<br/>

`# Create a ridge regressor: ridge`<br/>
`ridge = Ridge(normalize=True)`<br/>

`# Compute scores over range of alphas`<br/>
`for alpha in alpha_space:`<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`    # Specify the alpha value to use: ridge.alpha`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`    ridge.alpha = alpha`<br/>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`    # Perform 10-fold CV: ridge_cv_scores`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`    ridge_cv_scores = cross_val_score(ridge, X, y, cv = 10)`<br/>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`    # Append the mean of ridge_cv_scores to ridge_scores`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`    ridge_scores.append(np.mean(ridge_cv_scores))`<br/>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`    # Append the std of ridge_cv_scores to ridge_scores_std`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`    ridge_scores_std.append(np.std(ridge_cv_scores))`<br/>

`# Display the plot` <br/>
`display_plot(ridge_scores, ridge_scores_std)`