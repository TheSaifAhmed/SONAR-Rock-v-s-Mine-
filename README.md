# Logistic Regression Model for Predicting Rock and Mine
## Project Overview
This project involves constructing a Logistic Regression model to classify objects as either rock or mine based on sonar signal data. The model achieved an accuracy of 76.1% on the test dataset, showcasing its effectiveness in making accurate predictions.

## Dataset
The dataset used in this project is the Sonar dataset, which contains 208 samples of sonar signals bounced off metal cylinders (mines) and rocks. Each sample is described by 60 continuous features representing the energy of the sonar signal at various frequencies.

## Project Steps
### 1. Data Preprocessing
Loading the Dataset: The dataset is loaded into a Pandas DataFrame.
Exploratory Data Analysis (EDA): EDA is performed to understand the data distribution, check for missing values, and visualize the features.
### 2. Splitting the Dataset
The dataset is split into training and testing sets using an 80-20 split, providing 166 samples for training and 42 samples for testing.

### 3. Model Construction
A Logistic Regression model is constructed using the scikit-learn library. Logistic Regression is chosen due to its simplicity and effectiveness in binary classification problems.

### 4. Model Training and evaluation
The model is trained on the training set using the fit method. The model's performance is evaluated on the test set using accuracy as the metric. The model achieved an accuracy of 76.1% on the test data.

## Conclusion
This project demonstrates the application of Logistic Regression in a real-world scenario, providing valuable insights into the classification of sonar signals. The model's accuracy of 76.1% indicates its capability to make reliable predictions, and further tuning and feature engineering could potentially enhance its performance.
