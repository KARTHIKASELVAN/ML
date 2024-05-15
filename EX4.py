import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/FDSA PROGRAM/DATASET/tennisdata.csv")
print("The first 5 values of data are:\n", data.head())

# Obtain train data and train output
X = data.iloc[:, :-1]
print("\nThe first 5 values of train data are:\n", X.head())

y = data.iloc[:, -1]
print("\nThe first 5 values of train output are:\n", y.head())

# Convert categorical features to numerical values
le_outlook = LabelEncoder()
X['Outlook'] = le_outlook.fit_transform(X['Outlook'])

le_Temperature = LabelEncoder()
X['Temperature'] = le_Temperature.fit_transform(X['Temperature'])

le_Humidity = LabelEncoder()
X['Humidity'] = le_Humidity.fit_transform(X['Humidity'])

le_Windy = LabelEncoder()
X['Windy'] = le_Windy.fit_transform(X['Windy'])

print("\nNow the train data is:\n", X.head())

le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)

print("\nNow the train output is:\n", y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Initialize and train the classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Evaluate the classifier
accuracy = accuracy_score(classifier.predict(X_test), y_test)
print("Accuracy is:", accuracy)
