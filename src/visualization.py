import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df):
    for col in df.select_dtypes(include='number').columns:
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

def plot_categorical_counts(df, column):
    sns.countplot(x=column, data=df)
    plt.title(f'Counts of {column}')
    plt.show()