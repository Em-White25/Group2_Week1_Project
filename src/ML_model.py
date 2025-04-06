import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Function to load the dataset
def load_data(file_path):
    """Load the Titanic dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

# Function to clean the dataset by removing unwanted columns
def drop_unwanted_columns(df):
    """Drop unnecessary columns from the dataset."""
    columns_to_drop = ['name', 'home.dest', 'cabin', 'ticket', 'sibsp', 'parch', 'boat', 'body']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df

# Function to remove duplicate values
def remove_duplicates(df):
    """Remove duplicate rows from the dataset."""
    df.drop_duplicates(inplace=True)
    return df

# Function to fill missing age values based on the mean age per passenger class
def fill_missing_age(df, pclass_means_age):
    """Fill missing age values with the mean age per pclass."""
    def mean_age(cols):
        age, pclass = cols
        if pd.isnull(age):
            return int(round(pclass_means_age[pclass]))  # Round and convert to integer
        elif age < 1:
            return 1
        else:
            return int(round(age))  # Ensure age is also a whole number
    
    df['age'] = df[['age', 'pclass']].apply(mean_age, axis=1)
    return df

# Function to fill missing fare values based on the mean fare per passenger class
def fill_missing_fare(df, pclass_means_fare):
    """Fill missing fare values with the mean fare per pclass."""
    def mean_fare(cols):
        fare, pclass = cols
        if pd.isnull(fare):
            return int(round(pclass_means_fare[pclass]))  # Round and convert to integer
        else:
            return int(round(fare))  # Ensure fare is also a whole number

    df['fare'] = df[['fare', 'pclass']].apply(mean_fare, axis=1)
    return df

# Function to drop rows with missing values
def drop_missing_values(df):
    """Drop rows with missing values."""
    df.dropna(inplace=True)
    return df

# Function to clean the dataset (combined process)
def clean_data(file_path, output_path):
    """Load, clean, and save the dataset."""
    # Load dataset
    df_titanic = load_data(file_path)
    
    # Create a copy of the original dataset
    df_titanic_copy = df_titanic.copy()
    df_titanic_copy.index.rename("S/N", inplace=True)
    
    # Drop unwanted columns
    df_titanic_copy = drop_unwanted_columns(df_titanic_copy)
    
    # Remove duplicates
    df_titanic_copy = remove_duplicates(df_titanic_copy)
    
    # Compute mean age and fare for each pclass
    pclass_means_age = df_titanic_copy.groupby('pclass')['age'].mean().round().astype(int)
    pclass_means_fare = df_titanic_copy.groupby('pclass')['fare'].mean().round().astype(int)
    
    # Fill missing age and fare values
    df_titanic_copy = fill_missing_age(df_titanic_copy, pclass_means_age)
    df_titanic_copy = fill_missing_fare(df_titanic_copy, pclass_means_fare)
    
    # Drop rows with missing values (if any remain)
    df_titanic_copy = drop_missing_values(df_titanic_copy)
    
    # Save cleaned data
    df_titanic_copy.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    
    return df_titanic_copy

# Function to perform one-hot encoding
def encode_features(df):
    """One-hot encode categorical features: 'sex' and 'embarked'."""
    sex = pd.get_dummies(df['sex'], drop_first=True)
    embark = pd.get_dummies(df['embarked'], drop_first=True)
    
    # Concatenate the encoded features
    df = pd.concat([df, sex, embark], axis=1)
    
    # Drop original categorical columns
    df = df.drop(columns=['sex', 'embarked'], errors='ignore')
    
    return df

# Function for train-test split
def train_test_split_data(df):
    """Split data into training and testing sets."""
    X = df.drop('survived', axis=1)
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    return X_train, X_test, y_train, y_test

# Function to train a Random Forest model
def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=101)
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to tune hyperparameters using GridSearchCV
def grid_search_rf(X_train, y_train):
    """Perform grid search to find the best hyperparameters for the Random Forest model."""
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
    }

    rf_model = RandomForestClassifier(random_state=101)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-validation Accuracy: {grid_search.best_score_}")
    
    return grid_search.best_estimator_

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test data."""
    predictions = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

# Main function to clean the data, train the model, and evaluate it
def main():
    input_file = 'data/raw/Titanic dataset.csv'
    output_file = 'data/processed/Cleaned_Titanic_Dataset.csv'
    
    # Step 1: Clean the data
    cleaned_data = clean_data(input_file, output_file)
    
    # Step 2: Preprocess features
    cleaned_data = encode_features(cleaned_data)
    
    # Step 3: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split_data(cleaned_data)
    
    # Step 4: Train the Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    
    # Step 5: Tune the model with GridSearchCV
    best_rf_model = grid_search_rf(X_train, y_train)
    
    # Step 6: Evaluate the model on the test set
    evaluate_model(best_rf_model, X_test, y_test)

if __name__ == "__main__":
    main()
