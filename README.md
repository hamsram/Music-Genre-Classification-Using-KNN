#  Music Genre Classification Using KNN

A machine learning-based music genre classification project that uses the **K-Nearest Neighbors (KNN)** algorithm to predict music genres from extracted audio features. The model uses feature standardisation, hyperparameter tuning, and cross-validation to improve classification performance.

###  Key Features

- **Data Preprocessing:** Cleans the dataset and separates audio features from genre labels.
- **Label Encoding:** Converts categorical music genre labels into numerical representations.
- **Feature Standardisation:** Uses StandardScaler to normalise numerical features.
- **Genre Classification:** Implements KNN to classify music samples based on feature similarity.
- **Hyperparameter Tuning:** Uses GridSearchCV to identify the optimal number of neighbours.
- **Model Evaluation:** Applies Stratified K-Fold Cross-Validation to evaluate classification accuracy.
- **Prediction Export:** Generates genre predictions for unseen samples and exports them into a CSV file.

###  Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Classification Algorithm:** K-Nearest Neighbors (KNN)
- **Feature Scaling:** StandardScaler
- **Hyperparameter Tuning:** GridSearchCV
- **Model Validation:** Stratified K-Fold Cross-Validation
