from data_processing import load_data, clean_data, save_cleaned_data
from eda import show_basic_info, plot_correlation_matrix
from visualization import plot_distributions, plot_categorical_counts
from model import train_model
from datetime import datetime

def main():
    # Step 1: Load & clean data
    df = load_data('data/raw/Titanic dataset.csv')
    df_cleaned = clean_data(df)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_cleaned_data(df_cleaned, f'data/processed/cleaned_data_{timestamp}.csv')

    # Step 2: EDA & Visualization
    show_basic_info(df_cleaned)
    plot_correlation_matrix(df_cleaned)
    plot_distributions(df_cleaned)
    plot_categorical_counts(df_cleaned, column='survived')  

    # Step 3: Model training
    model = train_model(df_cleaned, target_column='survived')  

if __name__ == '__main__':
    main()