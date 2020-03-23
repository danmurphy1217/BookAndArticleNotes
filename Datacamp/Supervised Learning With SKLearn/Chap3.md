# How Good is a Model?

Use a confusion matrix w/ False Pos. and False Neg. for predicted vs. actual values. Example: <br/>
![Confusion Matrix](confusionmatrix.png)<br/>
Accuracy: Sum of the diagonal divided by the total sum of the matrix <br/>
Precision: True positives divided by true positives plus false positives <br/>
Recall: True positives divided by true positives plus false negatives <br/>
F1score: (2*precision*recall)/(precision+recall) <br/>

### Classification
`from sklearn.metrics import classification_report`<br/>
`from sklearn.metrics import confusion_matrix`<br/>
`# Create training and test set`<br/>
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state =42)`<br/>

`# Instantiate a k-NN classifier: knn`<br/>
`knn = KNeighborsClassifier(n_neighbors=6)`<br/>

`# Fit the classifier to the training data`<br/>
`knn.fit(X_train, y_train)`<br/>

`# Predict the labels of the test data: y_pred`<br/>
`y_pred = knn.predict(X_test)`<br/>

`# Generate the confusion matrix and classification report`<br/>
`print(classification_report(y_test, y_pred))`<br/>
`print(confusion_matrix(y_test, y_pred))`<br/>

### Log Reg Model <br/>
`# Import the necessary modules`<br/>
`from sklearn.linear_model import LogisticRegression`<br/>
`from sklearn.metrics import confusion_matrix, classification_report`<br/>

`# Create training and test sets`<br/>
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)`<br/>

`# Create the classifier: logreg`<br/>
`logreg = LogisticRegression()`<br/>

`# Fit the classifier to the training data`<br/>
`logreg.fit(X_train, y_train)`<br/>

`# Predict the labels of the test set: y_pred`<br/>
`y_pred = logreg.predict(X_test)`<br/>

`# Compute and print the confusion matrix and classification report`<br/>
`print(confusion_matrix(y_test, y_pred))`<br/>
`print(classification_report(y_test, y_pred))`<br/>

### Plot a ROC Curve <br/>
`# Import necessary modules`<br/>
`from sklearn.metrics import roc_curve`<br/>

`# Compute predicted probabilities: y_pred_prob`<br/>
`y_pred_prob = logreg.predict_proba(X_test)[:,1]`<br/>

`# Generate ROC curve values: fpr, tpr, thresholds`<br/>
`fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)`<br/>

`# Plot ROC curve`<br/>
`plt.plot([0, 1], [0, 1], 'k--')`<br/>
`plt.plot(fpr, tpr)`<br/>
`plt.xlabel('False Positive Rate')`<br/>
`plt.ylabel('True Positive Rate')`<br/>
`plt.title('ROC Curve')`<br/>
`plt.show()`<br/>

### AUC Computation (Area under ROC Curve)

`# Import necessary modules` <br/>
`from sklearn.metrics import roc_auc_score`<br/>
`from sklearn.model_selection import cross_val_score`<br/>

`# Compute predicted probabilities: y_pred_prob`<br/>
`y_pred_prob = logreg.predict_proba(X_test)[:,1]`<br/>

`# Compute and print AUC score`<br/>
`print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))`<br/>

`# Compute cross-validated AUC scores: cv_auc`<br/>
`cv_auc = cross_val_score(logreg, X, y, cv = 5, scoring = 'roc_auc')`<br/>

`# Print list of AUC scores`<br/>
`print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))`<br/>

### Hyperparameter Tuning w/ GridSearchCV<br/>

`# Import necessary modules`<br/>
`from sklearn.linear_model import LogisticRegression`<br/>
`from sklearn.model_selection import GridSearchCV`<br/>

`# Setup the hyperparameter grid`<br/>
`c_space = np.logspace(-5, 8, 15)`<br/>
`param_grid = {'C': c_space}`<br/>

`# Instantiate a logistic regression classifier: logreg`<br/>
`logreg = LogisticRegression()`<br/>

`# Instantiate the GridSearchCV object: logreg_cv`<br/>
`logreg_cv = GridSearchCV(logreg, param_grid, cv=5)`<br/>

`# Fit it to the data`<br/>
`logreg_cv.fit(X, y)`<br/>

`# Print the tuned parameters and score`<br/>
`print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))`<br/>
`print("Best score is {}".format(logreg_cv.best_score_))`<br/>

### Hyperparameter Tuning w/ RandomizedSearchCV<br/>

`# Import necessary modules`<br/>
`from scipy.stats import randint`<br/>
`from sklearn.tree import DecisionTreeClassifier`<br/>
`from sklearn.model_selection import RandomizedSearchCV`<br/>

`# Setup the parameters and distributions to sample from: param_dist`<br/>
`param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}`<br/>

`# Instantiate a Decision Tree classifier: tree`<br/>
`tree = DecisionTreeClassifier()`<br/>

`# Instantiate the RandomizedSearchCV object: tree_cv`<br/>
`tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)`<br/>

`# Fit it to the data`<br/>
`tree_cv.fit(X, y)`<br/>

`# Print the tuned parameters and score`<br/>
`print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))`<br/>
`print("Best score is {}".format(tree_cv.best_score_))`<br/>

### Hold-out set in practice: Classification<br/>

`# Import necessary modules`<br/>
`from sklearn.model_selection import train_test_split`<br/>
`from sklearn.linear_model import LogisticRegression` <br/>
`from sklearn.model_selection import GridSearchCV`<br/>

`# Create the hyperparameter grid`<br/>
`c_space = np.logspace(-5, 8, 15)`<br/>
`param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}`<br/>

`# Instantiate the logistic regression classifier: logreg`<br/>
`logreg = LogisticRegression()`<br/>

`# Create train and test sets`<br/>
`X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = .4)`<br/>

`# Instantiate the GridSearchCV object: logreg_cv`<br/>
`logreg_cv = GridSearchCV(logreg, param_grid, cv = 5)`<br/>

`# Fit it to the training data`<br/>
`logreg_cv.fit(X_train, y_train)`<br/>

`# Print the optimal parameters and best score`<br/>
`print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))`<br/>
`print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))`<br/>


### Hold-out set in practice: Regression

`# Import necessary modules`<br/>
`from sklearn.linear_model import ElasticNet`<br/>
`from sklearn.metrics import mean_squared_error`<br/>
`from sklearn.model_selection import train_test_split, GridSearchCV`<br/>


`# Create train and test sets`<br/>
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state = 42)`<br/>

`# Create the hyperparameter grid`<br/>
`l1_space = np.linspace(0, 1, 30)`<br/>
`param_grid = {'l1_ratio': l1_space}`<br/>

`# Instantiate the ElasticNet regressor: elastic_net`<br/>
`elastic_net = ElasticNet()`<br/>
`# Setup the GridSearchCV object: gm_cv`<br/>
`gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)`<br/>

`# Fit it to the training data`<br/>
`gm_cv.fit(X_train, y_train)`<br/>

`# Predict on the test set and compute metrics`<br/>
`y_pred = gm_cv.predict(X_test)`<br/>
`r2 = gm_cv.score(X_test, y_test)`<br/>
`mse = mean_squared_error(y_test, y_pred)`<br/>
`print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))`<br/>
`print("Tuned ElasticNet R squared: {}".format(r2))`<br/>
`print("Tuned ElasticNet MSE: {}".format(mse))<br/>`
