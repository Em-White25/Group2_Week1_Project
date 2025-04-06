# libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

def load_data(file_path):
    """Load cleaned Titanic dataset from csv."""
    return pd.read_csv(file_path)

def display_head(df, n=5):
    """Display the first few rows of the dataframe."""
    print(df.head(n))

def check_missing_values(df):
    """Check for missing values in the dataset."""
    print("\nMissing values:\n", df.isnull().sum())

def display_structure(df):
    """Display structure and summary statistics of the dataset."""
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())

def summary_statistics(df):
    """Generate summary statistics for selected numerical columns."""
    numerical_cols = ['age', 'fare', 'survived']
    summary_stats = df[numerical_cols].describe()
    print(summary_stats)

def plot_age_distribution(df):
    """Plot age distribution histogram."""
    plt.figure(figsize=(8,5))
    sns.histplot(df['age'], bins=30, kde=True, color='teal')
    plt.title("Age Distribution")
    plt.show()

def plot_fare_distribution(df):
    """Plot histogram of fare distribution."""
    plt.figure(figsize=(8,5))
    sns.histplot(df['fare'], bins=40, kde=True, color='purple')
    plt.title("Fare Distribution")
    plt.xlabel("Fare Amount")
    plt.ylabel("Frequency")
    plt.show()

def plot_fare_boxplot(df):
    """Plot boxplot of fare to identify outliers."""
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['fare'], color='orange')
    plt.title("Fare Distribution")
    plt.xlabel("Fare Amount")
    plt.show()

def plot_survival_pie_chart(df):
    """Plot pie chart showing survival proportions."""
    plt.figure(figsize=(5, 5))
    df['survived'].value_counts().plot.pie(
        autopct='%1.1f%%', 
        colors=["red", "green"],
        labels=["Did Not Survive", "Survived"],
        startangle=90
    )
    plt.title("Survival Rate")
    plt.ylabel("")
    plt.show()

def plot_fare_by_class(df):
    """Plot fare distribution across passenger classes."""
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='pclass', y='fare', data=df, palette='Oranges')
    plt.title("Fare Distribution by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare")
    plt.show()

def plot_pairplot_survival(df):
    """Plot pairplot to explore relationships between survival and key features."""
    sns.pairplot(df, hue='survived', vars=['age', 'fare', 'pclass'], palette='coolwarm')
    plt.suptitle("Pair Plot: Relationships Between Survival and Key Features", y=1.02)
    plt.show()

def plot_survival_by_class(df):
    """Plot survival rate by passenger class."""
    plt.figure(figsize=(7, 5))
    sns.barplot(x='pclass', y='survived', data=df, ci=None, palette="pastel", order=[1, 2, 3])
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Average Survival Rate")
    plt.show()

def plot_survival_by_gender(df):
    """Plot survival rate by gender."""
    plt.figure(figsize=(7, 5))
    sns.barplot(x='sex', y='survived', data=df)
    plt.title("Survival Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Survival Rate")
    plt.show()

def plot_feature_correlations(df):
    """Plot heatmap of feature correlations, excluding 'S/N' if present."""
    numeric_df = df.select_dtypes(include=[np.number])
    if 'S/N' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['S/N'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlations (Excluding S/N)")
    plt.show()

def perform_eda(input_path):
    """Run full exploratory data analysis pipeline."""
    df = load_data(input_path)

    display_head(df)
    check_missing_values(df)
    display_structure(df)
    summary_statistics(df)
    
    plot_age_distribution(df)
    plot_fare_distribution(df)
    plot_fare_boxplot(df)
    plot_survival_pie_chart(df)
    plot_fare_by_class(df)
    plot_pairplot_survival(df)
    plot_survival_by_class(df)
    plot_survival_by_gender(df)
    plot_feature_correlations(df)

if __name__ == "__main__":
    input_file = "../data/processed/titanic_cleaned.csv"
    perform_eda(input_file)
