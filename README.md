# Churn Prediction System


## Table of Contents
- [Features](#features)
- [Description](#description)
- [Dataset](#dataset)
- [Model](#model)
- [Frontend](#frontend)
- [Working](#working)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Requirements](#requirements)
- [License](#license)
- [Instruction](#instructions)

## Features
1. **User-Friendly Input**: Input customer data such as the number of service calls, voicemail plans, and voicemail messages.
2. **Churn Prediction**: Predicts customer churn based on the provided data.
3. **Data Visualization**: Pie charts, bar charts, and histograms for data analysis.
4. **CSV Export**: Download predictions and analyzed data in CSV format.
5. **Data Analysis**: Displays descriptive statistics, including mean, median, mode, and custom sorting options.
6. **Pagination**: View large datasets in manageable chunks.

## Description
The **Churn Prediction System** is a web application that predicts customer churn using a machine learning model. 
It features a user-friendly **Streamlit** interface for inputting customer data, generating predictions, and visualizing results. 
The project also offers comprehensive data analysis tools, including statistical insights, visualizations, and CSV download options.


## Dataset
The dataset used in the Churn Prediction System contains information about customers and their behaviors, with the goal of predicting whether a customer will churn (leave the service) or stay. The dataset includes various features that describe customer activity and attributes, which are used as inputs to train the machine learning model.

## Model

The **Backend Model** of the Churn Prediction System is responsible for the machine learning model training, evaluation, and deployment. It handles data preprocessing, model building, and the prediction process. The backend is implemented using Python and its various libraries, and it leverages a **Jupyter Notebook** (`Churn_Pred_model.ipynb`) for model training and evaluation.

### Key Components of the Backend Model:
1. **Data Preprocessing**:
   - In the backend, the data is cleaned and preprocessed. This includes handling missing values, scaling features, and reducing dimensionality using **Principal Component Analysis (PCA)**. 
   - Preprocessing steps ensure that the data is in the right format before being passed to the machine learning model for training.

2. **Model Training**:
   - The backend model is trained using the preprocessed data. Various machine learning algorithms such as **Logistic Regression**, **Random Forest**, or **Gradient Boosting** can be used for training.
   - The model is evaluated using performance metrics like accuracy, precision, recall, and F1-score, which are critical to ensure its prediction reliability.

3. **Model Export**:
   - Once the model is trained and validated, it is serialized using **

## Frontend

The **frontend** of the Churn Prediction System is built using **Streamlit**, providing an interactive and user-friendly interface for data input, predictions, visualizations, and data analysis.

### Key Features of the Frontend:
1. **User Input Form**: 
   - Users can input customer data for the following features:
     - **PC1**: Number of customer service calls.
     - **PC2**: Presence of a voicemail plan.
     - **PC3**: Number of voicemail messages.
   - The form ensures ease of use with validation for appropriate input ranges.

2. **Churn Prediction**:
   - Upon submission of input data, the system uses a trained machine learning model to predict whether the customer will churn or remain active.
   - Results are displayed directly on the interface.

3. **Data Visualization**:
   - The frontend provides several visualization options:
     - **Pie Chart**: Displays the distribution of churned vs. active customers.
     - **Bar Chart**: Highlights counts of churned and active customers.
     - **Histograms**: Show the distribution of input features (e.g., PC1, PC2, PC3).

4. **CSV Export**:
   - A dedicated option allows users to download the dataset, including predictions and other analysis results, as a CSV file.

5. **Analysis Tab**:
   - The frontend features an **Analysis Tab** for deeper exploration of the dataset. It provides:
     - **Sorting**: Sort dataset columns in ascending or descending order based on user selection.
     - **Filtering**: Filter data to focus on specific values or ranges.
     - **Pagination**: View data in manageable chunks for better readability.
     - **Descriptive Statistics**: Display statistical measures such as:
       - Mean
       - Median
       - Mode
       - Minimum and Maximum values
       - Quartile ranges (25%, 50%, and 75%)
       - Standard deviation
   - These features help users gain actionable insights from the data and refine their understanding of customer behavior.

6. **Session State**:
   - The app leverages **Streamlit's session state** to manage user inputs, predictions, and analysis across multiple interactions. This ensures a smooth experience without data loss.

---
### Working:
1. Open the app using:
   ```bash
   streamlit run Frontend.py
   ```
2. Input the required customer data into the form and click **Submit** to get predictions.
3. Explore the **Visualization Tab** to see churn distribution and feature trends.
4. Navigate to the **Analysis Tab** for sorting, filtering, and analyzing data, including viewing statistical measures.
5. Download the dataset with predictions as a CSV file for further use.

---

## Project Structure

The project directory is organized as follows:

- `Frontend.py`  - Streamlit frontend script for user interaction and prediction display
- `requirements.txt`  - Dependency file containing the libraries required to run the project
- `Churn.pkl`  - Trained machine learning model used for churn prediction
- `impute.joblib`  - Imputer for handling missing values in the data
- `scaler.joblib`  - Scaler for normalizing features before prediction
- `pca_df_transformed.joblib`  - PCA-transformed dataset used for reducing dimensionality
- `df_labels.joblib`  - Labels for prediction, used during the model training process
- `Churn_Pred_model.ipynb`  - Jupyter Notebook for training the backend model, data preprocessing, and evaluation
- `README.md`  - Project documentation that includes setup and usage instructions

---

## Installation

Here’s the **Requirements** section in Markdown format with each library listed on a separate line:

---

## Requirements

The `requirements.txt` includes the following libraries:

- `streamlit`
- `pandas`
- `joblib`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `numpy`

To install these dependencies, run:
```bash
pip install -r requirements.txt
```

---

Dataset

https://drive.google.com/file/d/1AIM_kGf1zzFex4BXYZPkypiJsGtkVA7u/view?usp=sharing

Model

https://drive.google.com/file/d/1qTy-W8yQPyUZ2GeUbwIF4Fb8inUs7EF1/view?usp=sharing

Frontend

https://drive.google.com/file/d/1Kis5CUkf0ed09n-638dHcC_xEUygbL6D/view?usp=sharing


1. Run the app:
   ```bash
   streamlit run Frontend.py
   ```

## License
This project is licensed under the MIT License. See the LICENSE file for details

---

### Instructions
1. Save this as `README.md` in the root directory of your project.
2. Ensure all required files (`Churn.pkl`, etc.) are included in your repository.
3. Upload the `README.md` and other project files to your GitHub repository.


