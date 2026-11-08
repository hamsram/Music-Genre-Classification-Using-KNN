# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

# Load the training dataset
train_csv_path = 'train.csv'
data = pd.read_csv(train_csv_path)

# Preprocess the data
data = data.drop(columns=['filename'])
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]  # Labels

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Set up Stratified K-Fold Cross-Validation
k = 15  # Number of folds (you can adjust this)
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize a list to store accuracy scores
accuracies = []

# Define a range of 'k' values to search
param_grid = {'n_neighbors': range(1, 21)}  # Range of k values (adjust as needed)

# Iterate through each fold for training and validation
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Create a K-NN model
    knn_model = KNeighborsClassifier()

    # Perform a grid search using cross-validation to find the best 'k'
    grid_search = GridSearchCV(knn_model, param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best 'k' value from the grid search
    best_k = grid_search.best_params_['n_neighbors']

    # Train the K-NN model with the best 'k' value
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)

    # Predict labels for the validation set
    y_val_pred = knn_model.predict(X_val)

    # Calculate accuracy for the validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    accuracies.append(accuracy)

# Calculate the mean accuracy across all folds
mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy Across Folds: {mean_accuracy}")

# Train the final K-NN model on the entire training dataset with the best 'k' value
final_knn_model = KNeighborsClassifier(n_neighbors=best_k)
final_knn_model.fit(X, y)

# Load the test dataset and make predictions
test_csv_path = 'test.csv'
test_data = pd.read_csv(test_csv_path)
test_ids = test_data['id']  # Assuming 'id' is the column containing IDs
test_data = test_data.drop(columns=['id'])  # Drop 'id' and 'filename' columns

# Normalize the test features
test_data = scaler.transform(test_data)

# Predict labels for the test dataset
test_predictions = final_knn_model.predict(test_data)

final_accuracy = np.mean(accuracies)
print(f"Final Accuracy: {final_accuracy}")

# Save the predictions to a submission.csv file
submission_df = pd.DataFrame({'id': test_ids, 'label': test_predictions})
submission_df.to_csv('music_genre_predictions.csv', index=False)
