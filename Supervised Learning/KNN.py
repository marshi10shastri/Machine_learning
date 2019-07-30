# Importing libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

# Loading the inbuilt breast cancer data
cancer_data = load_breast_cancer()
# This cancer data is in form of dictionary
print(cancer_data)

# Displaying the feature names and the target names in the data 
print(cancer_data.feature_names)
print(cancer_data.target_names)

# Splitting the dataset for training and testing purpose into 7 : 3 ratio
x_train, x_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size =0.3, random_state = 1)

# Creating K nearest neighbors model
model = KNeighborsClassifier(n_neighbors=1)

# Fitting the traing data into the model
model.fit(x_train, y_train)

# Getting predictions from our model
predictions = model.predict(x_test)

# Calculationg the accuracy of our model
accuracy = accuracy_score(y_test, predictions) * 100
print(accuracy)