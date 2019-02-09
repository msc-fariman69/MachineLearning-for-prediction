# I'm ashamed for explain this part!!!!!!!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

np.set_printoptions(threshold=np.inf)

# Read csv files as Data Frame
test_df = pd.read_csv('assets/Test.csv')
main_test_df = test_df
train_df = pd.read_csv('assets/Train.csv')

# Let's look at structure of our data
print('--------------------------------Structure of our Data--------------------------------')
print(train_df.info())
print()

# Now...
print('----------------------------------Describe our Data----------------------------------')
print(train_df.describe())
print()
# Visualize our data
print('-------------------------------Visualize Data ( See plot )---------------------------')
train_df.hist(bins=50, figsize=(20, 15))
plt.show()
print()

# Um, our data contains none numeric values, therefore, i have to clean it.
# let's change it to Data Frame with none numeric values
# Step one for clean data.
test_df['residence_area_type'] = test_df['residence_area_type'].astype('category')
test_df['sourcing_channel'] = test_df['sourcing_channel'].astype('category')
train_df['residence_area_type'] = train_df['residence_area_type'].astype('category')
train_df['sourcing_channel'] = train_df['sourcing_channel'].astype('category')

# Step two for clean data. at this part, i retrieve those columns
category_columns = test_df.select_dtypes(['category']).columns

# Step three for clean data. convert to numeric
test_df[category_columns] = test_df[category_columns].apply(lambda x: x.cat.codes)
train_df[category_columns] = train_df[category_columns].apply(lambda x: x.cat.codes)

# No! our data is imbalanced in class
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
# Get Balance Data frame
under_sample = train_df.loc[under_sample_indices]

# Convert Data frame to Numpy Array for using in Algorithm
test_features = test_df.values
train = under_sample.values

# Split train array to label and features array
train_features = train[:, 0:11]
train_labels = train[:, 11:]

# Blimey, our data contains NAN value. How would! How would it be! anyway... lets get rid of them...
# x = np.any(np.isnan(train_features))
# y = np.all(np.isfinite(train_features))
train_features[np.isnan(train_features)] = 0
test_features[np.isnan(test_features)] = 0

# Before use... i should flat label array
labels = np.array(train_labels).flatten()

# It's time to predict - use Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(train_features, labels)
test_prediction = lin_reg.predict(test_features)

print('---------------------------Generating result on new cvs file-------------------------------')
main_test_df.insert(11,'prediction_result', test_prediction)
main_test_df.to_csv('assets/Test_with_result.csv')
print('--------------------------------------file Generated---------------------------------------')

print('------------------------------------Prediction Result--------------------------------------')
print(test_prediction.shape)
print(test_prediction)
print()

print('-----------------------------------------Accuracy------------------------------------------')
prediction = lin_reg.predict(train_features)
lin_mse = mean_squared_error(labels, prediction)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
print()

print('----------------------------------------Accuracy CV----------------------------------------')
lin_scores = cross_val_score(lin_reg, train_features, labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Scores:", lin_rmse_scores)
print("Mean:", lin_rmse_scores.mean())
print("Standard deviation:", lin_rmse_scores.std())
print()

# print('-----------------------------------Accuracy on predictions---------------------------------')
# final_mse = mean_squared_error(test_labels, prediction)
# final_rmse = np.sqrt(final_mse)
# print()

# So far so good but...
# Please note that i can do some work for tune model for example :
# Visualize data in detail
# Calculate correlation matrix
# Combine attribute
# Transforming on data
# More cleaning data
# Feature scaling
# Search for getting best Hyper Parameter
# Use other algorithms

print('-----------------------Designed by Mehdi Etaati---------------------------')


