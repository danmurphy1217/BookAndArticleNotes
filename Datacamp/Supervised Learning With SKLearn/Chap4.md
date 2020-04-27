# Sklearn Pipelines

Boxplots are particularly useful for visualizing categorical features.

`# Import pandas`
`import pandas as pd`

`# Read 'gapminder.csv' into a DataFrame: df`
`df = pd.read_csv('gapminder.csv')`

`# Create a boxplot of life expectancy per region`
`df.boxplot('life','Region', rot=60)`

`# Show the plot`
`plt.show()`

code for a boxplot with the gapminder data.


This code below is used to create a dummy variable. The second call of get_dummies() is particularly useful because it uses drop_first=True, which will remove the category from which you are creating the dummies!<br>
`# Create dummy variables: df_region`
`df_region = pd.get_dummies(df)`

`# Print the columns of df_region`
`print(df_region.columns)`

`# Create dummy variables with drop_first=True: df_region`
`df_region = pd.get_dummies(df, drop_first=True)`

`# Print the new columns of df_region`
`print(df_region.columns)`

The following code uses a Ridge regressor to perform 5-fold cross validation and prints out the ridge cross validation scores. Lasso will penalize the model for the sum of the abosolute value of the weights. Ridge will take this a step furhter to penalize the model for the sum of the squared value of the weights. <br>

`# Import necessary modules`
`from sklearn.model_selection import cross_val_score`
`from sklearn.linear_model import Ridge`

`# Instantiate a ridge regressor: ridge`
`ridge = Ridge(alpha=0.5, normalize=True)`

`# Perform 5-fold cross-validation: ridge_cv`
`ridge_cv = cross_val_score(ridge, X, y, cv=5)`

`# Print the cross-validated scores`
`print(ridge_cv, X, y)`

use Imputer() to transform data in a given way. This is particularly helpful when we need to handle missing values.

The following code is used to remove '?' from the data and make them NaN using np.nan. Additionally, we then drop the na values. <br>

`# Convert '?' to NaN`
`df[df == '?'] = np.nan`

`# Print the number of NaNs`
`print(df.isnull().sum())`

`# Print shape of original DataFrame`
`print("Shape of Original DataFrame: {}".format(df.shape))`

`# Drop missing values and print shape of new DataFrame`
`df = df.dropna()`

`# Print shape of new DataFrame`
`print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))`

Scikit Learn pipelines are useful for simpifying workflow. The following is an example of setting one up: <br>

`# Import the Imputer module`
`from sklearn.preprocessing import Imputer`
`from sklearn.svm import SVC`

`# Setup the Imputation transformer: imp`
`imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)`

`# Instantiate the SVC classifier: clf`
`clf = SVC()`

`# Setup the pipeline with the required steps: steps`
`steps = [('imputation', imp),('SVM', clf)]`

This code above sets up our list of steps (in tuples) for the Pipeline. Now, we will instantiate the pipeline and build out the model.<br/>

`# Import necessary modules`
`from sklearn.preprocessing import Imputer`
`from sklearn.pipeline import Pipeline`
`from sklearn.svm import SVC`

`# Setup the pipeline steps: steps`
`steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),('SVM', SVC())]`

`# Create the pipeline: pipeline`
`pipeline = Pipeline(steps)`

`# Create training and test sets`
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42)`

`# Fit the pipeline to the train set`
`pipeline.fit(X_train, y_train)`

`# Predict the labels of the test set`
`y_pred = pipeline.predict(X_test)`

`# Compute metrics`
`print(classification_report(y_test, y_pred))`


classification_report from sklearn.metrics is also particularly useful, as it shows you the precision, recall, and f1-score. 


Below, the code shows us how to scale our data (AKA normalization). This is especially important when using a KNN model, because KNN makes predictions based on the distance between characterstics in our data points. 

`# Import scale`
`from sklearn.preprocessing import scale`

`# Scale the features: X_scaled`
`X_scaled = scale(X)`

`# Print the mean and standard deviation of the unscaled features`
`print("Mean of Unscaled Features: {}".format(np.mean(X)))`
`print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))`

`# Print the mean and standard deviation of the scaled features`
`print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))`
`print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))`

Below, this code chunk creates a pipeline for our scaler and model, fits the pipeline to our training data, and then compares the accuracy between the scaled and unscaled models.

`# Import the necessary modules`
`from sklearn.preprocessing import StandardScaler`
`from sklearn.pipeline import Pipeline`

`# Setup the pipeline steps: steps`
`steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier())]`

`# Create the pipeline: pipeline`
`pipeline = Pipeline(steps)`

`# Create train and test sets`
`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = .7)`

`# Fit the pipeline to the training set: knn_scaled`
`knn_scaled = pipeline.fit(X_train, y_train)`



`# Instantiate and fit a k-NN classifier to the unscaled data`
`knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)`

`# Compute and print metrics`
`print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))`
`print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))`

