# I'm ashamed for explain this part!!!!!!!
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
import sklearn.model_selection as model
from sklearn.metrics import confusion_matrix, f1_score

np.set_printoptions(threshold=np.inf)

# read csv files as DataFrame
test_df = pd.read_csv('assets/Test.csv')
train_df = pd.read_csv('assets/Train.csv')

# Ooooops, your data contains non-numeric values, therefore, i have to clean it. let's change it to Dataset with not numeric values
test_df['residence_area_type'] = test_df['residence_area_type'].astype('category')
test_df['sourcing_channel'] = test_df['sourcing_channel'].astype('category')
train_df['residence_area_type'] = train_df['residence_area_type'].astype('category')
train_df['sourcing_channel'] = train_df['sourcing_channel'].astype('category')

# step two for clean data. at this part, i retrieve those columns
category_columns = test_df.select_dtypes(['category']).columns

# convert to numeric
test_df[category_columns] = test_df[category_columns].apply(lambda x: x.cat.codes)
train_df[category_columns] = train_df[category_columns].apply(lambda x: x.cat.codes)

# no! your data is imbalanced in class
# Find Number of samples which are Fraud
no_frauds = len(train_df[train_df['renewal'] == 1])
# Get indices of non fraud samples
non_fraud_indices = train_df[train_df['renewal'] == 0].index
# Random sample non fraud indices
random_indices = np.random.choice(non_fraud_indices, non_fraud_indices.shape, replace=False)
# Find the indices of fraud samples
fraud_indices = train_df[train_df['renewal'] == 1].index
# Concat fraud indices with sample non-fraud ones
under_sample_indices = np.concatenate([fraud_indices, random_indices])
# Get Balance Dataframe
under_sample = train_df.loc[under_sample_indices]

# convert Datafarme to Numpy Array for using in Algorithm
test_features = test_df.values
train = under_sample.values

# split train and test array to label and features array
train_features = train[:, 0:11]
train_labels = train[:, 11:]

# print arrays
# print('------------------------------------Train-----------------------------------')
# print()
# print('-----------------------------------Features---------------------------------')
# print(train_features)
# print('------------------------------------Labels----------------------------------')
# print(train_labels)
# print()
# print()
# print()
# print('------------------------------------Test-----------------------------------')
# print()
# print('-----------------------------------Features---------------------------------')
# print(test_features)


# blimey, your data contains NAN value. How would! How would it be! anyway... lets get rid of them...
# x = np.any(np.isnan(train_features))
# y = np.all(np.isfinite(train_features))
train_features[np.isnan(train_features)] = 0
test_features[np.isnan(test_features)] = 0

# for review
# print(train_features.shape)
# print(test_features.shape)

# Before use... i should flat label array
labels = np.array(train_labels).flatten()

# again... for review
# print(train_features.shape)
# print(test_features.shape)

# Blah Blah Blah...
# it's time to define SVM CLASSIFIER
svm_clf = LinearSVC(random_state=0, tol=1e-5)
# train phase
svm_clf.fit(train_features, labels)
# prediction phase
predicted = svm_clf.predict(test_features)
number_of_zero = predicted[predicted == 0]
number_of_one = predicted[predicted == 1]
# print(svm_clf.n_iter_)
print('-----------------------------------Prediction-------------------------------------')
print()
print(number_of_one.shape)
print(number_of_zero.shape)
print()
print()
print('------------------------------------Measures-------------------------------------')
# finally it's time to measure accuracy
# Cross Validation Accuracy
print()
print('-------------------------------------Accuracy------------------------------------')
accuracy = model.cross_val_score(svm_clf, train_features, labels, cv=3, scoring="accuracy")
print(accuracy)
# Confusion Matrix
train_prediction = model.cross_val_predict(svm_clf, train_features, labels, cv=3)
cm = confusion_matrix(labels, train_prediction)
print('-------------------------------------Accuracy------------------------------------')
print(cm)
fs = f1_score(labels, train_prediction)
print('-------------------------------------F1-Score------------------------------------')
print(fs)
print()
print('-------------------------------Designed by Mehdi Etaati--------------------------')
# this is it
