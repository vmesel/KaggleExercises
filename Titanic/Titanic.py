import pandas as pd
#Import the Numpy library
import numpy as np
#Import 'tree' from scikit-learn library
from sklearn import tree

#train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train_url = open("/work/users/vinicius/GitHub/KaggleExercises/Titanic/train.csv", "r")
train = pd.read_csv(train_url, sep=",")
test_url = open("/work/users/vinicius/GitHub/KaggleExercises/Titanic/test.csv", "r")
# #"http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test# .csv"
test = pd.read_csv(test_url, sep=",")

# Rose vs Jack, or Female vs Male TASK
"""
print(train.Survived.value_counts())
print(train["Survived"].value_counts(normalize = True))
print(train["Survived"][train["Sex"] == 'male'].value_counts())
print(train["Survived"][train["Sex"] == 'female'].value_counts())
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))
"""
# Does age play a role? TASK
"""
# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] > 18] = 0
train["Child"][train["Age"] == 18] = 0


# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))
"""
# First Prediction TASK
"""
test_one = test #.copy()

# Initialize a Survived column to 0
test_one["Survived"] = float(0)
test_one["Survived"][test_one["Sex"] == "female"] = 1
# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`

print(test_one["Survived"])
print(test_one.Survived)
"""

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))