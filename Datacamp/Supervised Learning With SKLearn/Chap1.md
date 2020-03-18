# **Overview**

### Machine Learning: The art and science of giving computers the ability to learn from data without being explicitly programmed
### Labels Present: Supervised Learning
### Unlabelled Data: Unsupervised Learning

# **First few videos**
### Supervised Learning: There are predictor variables/features and a target variable. The aim is to build a model that is able to predict the target variable. 
### Classification: Target variable consists of categories
### Regression: Target variable is continuous
### Features = Predictor Variable = Independent Variable
### Samples are in rows, features are in columns!
### dataset.target_names() will return the columns in the dataset

# **First, focus on EDA**
`df.head()` will return the first 5 rows of each column in the dataset
`df.info()` will give the number of columns and the data type for each column
`df.describe()` will give the count, mean, std, min, 25%, 50%, 75%, and max for each column
sns.countplot() takes four args: x-axis data, hue, data, and palette. 

# **K Nearest Neighbors**
KNN words to predict the label of any data point by looking at the 'k' nearest data points. It creates a set of decision boundaries. Training the knn model to fit the data is done using the `.fit()` method. The `.predict()` method is used to predict the labels of new data. <br />
Fitting a classifier with KNN: <br/>
`from sklearn.neighbors import KNeighborClassifier`<br />
`knn = KNeighborClassifier(n_neighbors = 6)` where n_neighbors specifies the number of neighbors <br />
We can use `knn.fit(iris['data'], iris['target'])` to fit the knn model to the dataset. In this scenario we use the iris dataset. <br />
SciKit Learn requires the data is an np array or a pandas df. The features must also be continous and not categories like male or female. The target sub-df must have the same number of rows as the features data and **only** one column. <br />
# **congressional voting dataset:** <br />
`from sklearn.neighbors import KNeighborClassifier`<br />
`# create arrays for the features and response variable`<br />
`y = df['party'].values`<br />
`X = df.drop('party', axis =1).values`<br />
`# create a k-NN classifier with 6 neighbors`<br />
`knn = KNeighborsClassifier(n_neighbors = 6)`<br />
` #  fit  the classifier to the data`<br />
`knn.fit(X, y)` <br/>
# **predictions:** <br />
`# Import KNeighborsClassifier from sklearn.neighbors`<br />
`from sklearn.neighbors import KNeighborsClassifier `<br />
`# Create arrays for the features and the response variable`<br />
`y = df['party'].values`<br />
`X = df.drop('party', axis=1).values`<br />
`# Create a k-NN classifier with 6 neighbors: knn`<br />
`knn = KNeighborsClassifier(n_neighbors = 6)`<br />
`# Fit the classifier to the data`<br />
`knn.fit(X, y)` <br />
`# **Predict the labels for the training data X** `<br />
`y_pred = knn.predict(X)`<br />
`# Predict and print the label for the new data point X_new`<br />
`new_prediction = knn.predict(X_new)`<br />
`print("Prediction: {}".format(new_prediction))`<br />
# **Measuring Performance**
We will do this with training and testing dataset splits. For example, we will split the whole dataset into X_train, X_test, y_train, and y_test datasets. The test size can be specified. <br/>
`# Import necessary modules` <br/>
`from sklearn import datasets`<br/>
`import matplotlib.pyplot as plt`<br/>
`# Load the digits dataset: digits`<br/>
`digits = datasets.load_digits()`<br/>
`# Print the keys and DESCR of the dataset`<br/>
`print(digits.keys())`<br/>
`print(digits.DESCR)`<br/>
`# Print the shape of the images and data keys`<br/>
`print(digits.images.shape)`<br/>
`print(digits.data.shape)`<br/>
`# Display digit 1010`<br/>
`plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')`<br/>
`plt.show()`<br/>

# Splitting data into test and train
`# Import necessary modules`<br/>
`from sklearn.neighbors import KNeighborsClassifier`<br/>
`from sklearn.model_selection import train_test_split`<br/>
`# Create feature and target arrays`<br/>
`X = digits.data`<br/>
`y = digits.target`<br/>
`# Split into training and test set`<br/>
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42, stratify= y)`<br/>
`# Create a k-NN classifier with 7 neighbors: knn`<br/>
`knn = KNeighborsClassifier(n_neighbors = 7)`<br/>
`# Fit the classifier to the training data`<br/>
`knn.fit(X_train, y_train)`<br/>
`# Print the accuracy`<br/>
`print(knn.score(X_test, y_test))`<br/>

# Plotting Overfitting and Underfitting
`# Setup arrays to store train and test accuracies`<br/>
`neighbors = np.arange(1, 9)`<br/>
`train_accuracy = np.empty(len(neighbors))`<br/>
`test_accuracy = np.empty(len(neighbors))`<br/>
`# Loop over different values of k`<br/>
`for i, k in enumerate(neighbors):`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`# Setup a k-NN Classifier with k neighbors: knn`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`knn = KNeighborsClassifier(n_neighbors=k)`<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`# Fit the classifier to the training data`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`knn.fit(X_train, y_train)`<br/>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`#Compute accuracy on the training set`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`train_accuracy[i] = knn.score(X_train, y_train)`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`#Compute accuracy on the testing set`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`test_accuracy[i] = knn.score(X_test, y_test)`<br/>

`# Generate plot`<br/>
`plt.title('k-NN: Varying Number of Neighbors')`<br/>
`plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')`<br/>
`plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')`<br/>
`plt.legend()`<br/>
`plt.xlabel('Number of Neighbors')`<br/>
`plt.ylabel('Accuracy')`<br/>
`plt.show()`<br/>
