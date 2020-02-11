1. Loading data in SkLearn:
	
Data must be numeric and stored as a numpy array or scipy sparse matrix. Pandas Dfs can also be converted to numeric arrays.

Example:

import numpy as np 
X = np.random.random((10,5))
y= np.array(['M', 'M', 'F', 'F;', 'F', 'M'])

2. Preprocessing the data:

	A. Standardization:

		from sklearn.Preprocessing import StandardScaler

		scaler= StandardScaler().fit(X_train)
		standardized_X = scaler.transform(X_test)

	B. Normalization:

		from sklearn.Preprocessing import Normalizer

		scaler = Normalizer().fit(X_train)

		normalized_X = scaler.transform(X_train)

		normalized_X_test = scaler.transform(X_test)

	C. Binarization:

		from sklearn.Preprocessing import Binarizer
		
		binarizer = Binarizer(threshold=0..0).fit(X)

		binary_x = binarizer.transform(X)

3. Encoding the Categorical Variables:

