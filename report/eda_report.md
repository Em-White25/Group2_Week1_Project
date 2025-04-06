**Exploratory Data Analysis (EDA) Report**
*Overview*
This report outlines the exploratory analysis conducted on the cleaned Titanic dataset. The goal was to uncover patterns, understand feature relationships, and identify factors influencing survival.

*Key EDA Steps*
Loaded Cleaned Dataset: Used the processed Cleaned Titanic dataset cleaned by the Data Cleaning team.

Inspected Data Structure: Explored the dataset shape, column types, and missing values to ensure data integrity.

Visualized Survival by Gender:

Plotted average survival rate by gender to highlight disparity in survival rates (with females more likely to survive).

Generated Correlation Heatmap:

Focused on key features: pclass, age, fare, sex_female, sex_male, and survived.

Observed moderate positive correlation between sex_female and survived, and negative correlation with pclass.

*Insights*
Gender and Survival: Women had a higher survival rate than men.

Passenger Class: Lower class (higher pclass value) correlated with lower survival rates.

Fare: Higher fares slightly correlated with higher chances of survival.

*Output*
The updated EDA notebook (eda_notebook.ipynb) and supporting script (src/eda.py) have been committed and are ready for collaboration with the modeling team.