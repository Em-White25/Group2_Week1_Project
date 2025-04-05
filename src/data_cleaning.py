import pandas as pd
import numpy as np

def load_data(file_path):
    """Load raw dataset from CSV."""
    return pd.read_csv(file_path)

def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()

def drop_unwanted_columns(df, columns):
    """Drop specified columns if they exist."""
    return df.drop(columns=columns, errors='ignore')

def encode_gender(df):
    """Encode 'sex' column: male -> 0, female -> 1."""
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    return df

def fill_missing_age(df):
    """Fill missing 'age' with mean age per 'pclass'."""
    df['age'] = df.groupby('pclass')['age'].transform(lambda x: x.fillna(np.ceil(x.mean())))
    return df

def fill_missing_fare(df):
    """Fill missing 'fare' with mean fare per 'pclass'."""
    df['fare'] = df.groupby('pclass')['fare'].transform(lambda x: x.fillna(x.mean()))
    return df

def fill_missing_embarked(df):
    """Fill missing 'embarked' values with mode."""
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    return df

def round_fare(df):
    """Round fare values to 2 decimal places."""
    df['fare'] = df['fare'].round(2)
    return df

def clean_data(input_path, output_path):
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = drop_unwanted_columns(df, ['name', 'home.dest', 'cabin', 'ticket', 'sibsp', 'parch', 'boat', 'body'])
    df = encode_gender(df)
    df = fill_missing_age(df)
    df = fill_missing_fare(df)
    df = fill_missing_embarked(df)
    df = round_fare(df)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "../data/raw/Titanic dataset.csv"
    output_file = "../data/processed/titanic_cleaned.csv"
    clean_data(input_file, output_file)
