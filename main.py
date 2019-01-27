import pandas as pd

test_df = pd.read_csv('assets/Test.csv')
train_df = pd.read_csv('assets/Train.csv')

test_df['residence_area_type'] = test_df['residence_area_type'].astype('category')
test_df['sourcing_channel'] = test_df['sourcing_channel'].astype('category')
train_df['residence_area_type'] = train_df['residence_area_type'].astype('category')
train_df['sourcing_channel'] = train_df['sourcing_channel'].astype('category')

category_columns = test_df.select_dtypes(['category']).columns

test_df[category_columns] = test_df[category_columns].apply(lambda x: x.cat.codes)
train_df[category_columns] = train_df[category_columns].apply(lambda x: x.cat.codes)

test_features = test_df.values
train = train_df.values

train_features = train[:, 0:10]
train_labels = train[:, 11:]
print('------------------------------------Train-----------------------------------')
print()
print('-----------------------------------Features---------------------------------')
print(train_features)
print('------------------------------------Labels----------------------------------')
print(train_labels)
print()
print()
print()
print('------------------------------------Test-----------------------------------')
print()
print('-----------------------------------Features---------------------------------')
print(test_features)
