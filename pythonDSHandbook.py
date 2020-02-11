Supervised learning: Models that predict labes based on training data

Classification: Models that predict labels as two or more discrete categories

 Regression: Models that predict continous labels

Unsupervised Learning: Models that identify structure in unlabeled data

Clustering: Models that detect and identify distinct groups in the data

Dimensionality reduction: Models that detect and identify lower-dimensional structure in higher-dimensional data

In SKLearn, it is best practice to create a matrix of your independent variables and a separate vector (a matrix with one column) for your 
dependent variable (the variable youre trying to predict new values of). This can be accomplished with the following code:
	

	indepen_Vars = data.drop('name of col with independent data', axis = 1) # axis = 1 ensures you're looking at columns, not rows (rows are axis =0)
	indepen_Vars.shape #useful for ensuring the data was dropped


	depend_var = data['name of col with independent data']
	depend_var.shape #useful for checking the dimensions

	
