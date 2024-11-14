

# Customer Churn Prediction

This project focuses on predicting customer churn based on various features. The goal is to use a machine learning model to classify customers who are likely to churn. The code includes data preprocessing, feature engineering, model training, and evaluation to assess the model's effectiveness.

## Project Structure

- **Data Preprocessing**: Cleans and transforms the dataset, handling missing values, encoding categorical variables, and scaling numerical features.
- **Feature Engineering**: Creates relevant features to improve the predictive capability of the model.
- **Model Training**: Fits a machine learning model to predict customer churn without additional hyperparameter tuning.
- **Evaluation**: Uses evaluation metrics such as accuracy, precision, recall, and F1-score to gauge model performance.

## Requirements

- Python 3.8 or above
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

## Usage

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   Open `Churn_pred.ipynb` in Jupyter Notebook and run each cell sequentially to preprocess the data, train the model, and evaluate its performance.

## Code Walkthrough

1. **Data Loading**: Loads the dataset with customer information.
2. **Data Preprocessing**: Cleans and preprocesses the dataset, including handling missing values and scaling.
3. **Feature Engineering**: Develops relevant features to boost model performance.
4. **Model Training**: Trains a selected machine learning model.
5. **Evaluation**: Uses metrics to assess model performance and identify areas for improvement.

## Results

The model provides a solid baseline for predicting customer churn, with results assessed through F1-score and other metrics.

## Future Improvements

Possible enhancements include:
- Exploring additional data features
- Experimenting with other machine learning models



