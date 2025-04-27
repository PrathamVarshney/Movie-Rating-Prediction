# 🎬 **Movie Rating Prediction using IMDb Movies Dataset**

This project leverages machine learning techniques to predict movie ratings based on various features from the **IMDb Movies India** dataset. The goal is to predict the rating of a movie based on attributes like genre, director, actors, duration, and votes.

---

## 📋 **Table of Contents**
1. [Project Description]
2. [Technologies Used]
3. [Dataset]
4. [Steps Involved]
5. [Model Evaluation]
6. [Running the Code]

---

## 📜 **Project Description**
In this project, we explore the **IMDb Movies India dataset** and train a **Random Forest Regressor** model to predict movie ratings. The dataset contains various attributes, such as movie genre, director, actors, duration, and votes. The model's performance is evaluated using common regression metrics.

**Objective**: Predict movie ratings based on the movie's features.

### Key Features:
- **Genre**: Genre of the movie (e.g., Comedy, Drama)
- **Director**: Director of the movie
- **Actor 1, Actor 2, Actor 3**: Main actors
- **Duration**: Duration of the movie in minutes
- **Votes**: Number of votes received
- **Year**: Release year of the movie
- **Rating**: Movie rating (target variable)

---

## 🛠️ **Technologies Used**
- **Python**: The programming language used for data processing and model building.
- **pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and handling arrays.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning, including model building and evaluation.

---

## 📊 **Dataset**
The **IMDb Movies India dataset** (CSV file) contains detailed information about movies, such as ratings, genres, directors, actors, and other metadata. The dataset undergoes preprocessing to handle missing values, encode categorical variables, and create additional features.

**Columns in the Dataset**:
- `Name`: Movie name
- `Year`: Release year
- `Genre`: Genre of the movie
- `Director`: Director of the movie
- `Actor 1`, `Actor 2`, `Actor 3`: Main actors
- `Votes`: Number of votes
- `Duration`: Movie duration (in minutes)
- `Rating`: Movie rating (target variable)

---

## 🔄 **Steps Involved**

1. **Data Loading & Exploration**:
   - Load the dataset using `pandas`.
   - Perform basic exploration: preview the first few rows, check column names, find missing values, and compute summary statistics.

2. **Data Preprocessing**:
   - Handle missing values by filling them with the median for numerical columns and "Unknown" for categorical columns.
   - Clean the `Year` column to extract only the year value, and convert all numeric columns to their appropriate types.

3. **Feature Engineering**:
   - Create additional features such as:
     - `Director_Success`: Average rating of movies by each director.
     - `Genre_Avg_Rating`: Average rating of movies in each genre.

4. **Label Encoding**:
   - Encode categorical features (e.g., `Genre`, `Director`, `Actor 1`, `Actor 2`, `Actor 3`) using `LabelEncoder` from Scikit-learn.

5. **Model Building**:
   - Split the data into training and testing sets (80% for training, 20% for testing).
   - Train a **Random Forest Regressor** model on the training data.

6. **Model Evaluation**:
   - Evaluate the model using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**.

---

## 📊 **Model Evaluation**

The model's performance is evaluated with the following metrics:
- **Mean Absolute Error (MAE)**: The average of absolute differences between actual and predicted values.
- **Root Mean Squared Error (RMSE)**: The square root of the average of squared differences between actual and predicted values.
- **R² Score**: Indicates the proportion of variance in the target variable that the model can explain.

**Example Output**:
--**Random Forest Mean Absolute Error (MAE): 0.33
--**Random Forest Root Mean Squared Error (RMSE): 0.61
--**Random Forest R² Score: 0.62

---

## ⚡ **Running the Code**

### 1. **Clone the Repository** or Download the dataset:
   - Save the **IMDb Movies India.csv** file in your working directory.

### 2. **Install Required Libraries**:
   Run the following command to install the necessary libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn

### python movie_rating_prediction.py

## 🏆 Conclusion
This project demonstrates how to:
--**Preprocess data, handle missing values, and engineer features.
--**Build and train a Random Forest Regressor model for movie rating prediction.
--**Evaluate the model's performance using common regression metrics.
