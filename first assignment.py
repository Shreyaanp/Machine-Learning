import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.feature_selection import chi2

dataset = pd.read_csv('./Employee dataset IBM kaggle/employee_data.csv')

# Displaying the first few rows of the dataset
print(dataset.head())

# Finding Missing Data
missing_data = dataset.isnull().sum()
print("\nMissing Data:")
print(missing_data)

# Automatically detecting categorical columns
categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
print("\nDetected Categorical Columns:")
print(categorical_columns)

# Encoding Categorical Data
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Splitting the dataset into the training set and test set
X = dataset.drop('Attrition', axis=1).values  # Excluding the 'Attrition' column as it seems to be the target variable
y = dataset['Attrition'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("\nTraining set:")
print(X_train)
print(y_train)

print("\nTest set:")
print(X_test)
print(y_test)

# Assignment 2 - feature scaling

sc = StandardScaler()
Xsc_train = sc.fit_transform(X_train)
Xsc_test = sc.transform(X_test)

print("\nTraining set after feature scaling:")
print(Xsc_train)

print("\nTest set after feature scaling:")
print(Xsc_test)

# perform information gain on the feature scaling
mutual_info = mutual_info_classif(Xsc_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = dataset.drop('Attrition', axis=1).columns
mutual_info.sort_values(ascending=False)
mutual_info.plot.bar(figsize=(20, 8))
plt.show()

#perform chi square test on the feature scaling
chi_scores = chi2(abs(Xsc_train), y_train)
p_values = pd.Series(chi_scores[1], index=X_train.columns)
p_values.sort_values(ascending=False, inplace=True)
p_values.plot.bar(figsize=(20, 8))
plt.show()


# apply pearson corelation with the dataset
corr = dataset.corr(method='pearson')
print("\nPearson Correlation:")
print(corr)
# Heatmap for the above process
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()


