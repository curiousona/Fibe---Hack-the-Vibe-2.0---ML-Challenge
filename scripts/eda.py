import pandas as pd


# Load dataset with a specific encoding
train_df = pd.read_csv('../data/train.csv', encoding='ISO-8859-1')
print("Train Data Shape:", train_df.shape)


# Check the data structure
print(train_df.head())


# Check for missing values
print(train_df.isnull().sum())


# Check category distribution
print(train_df['target'].value_counts())


