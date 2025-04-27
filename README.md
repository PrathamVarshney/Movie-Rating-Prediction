Movie Rating Prediction using IMDb Movies Dataset
This project uses machine learning techniques to predict movie ratings based on various features from the IMDb Movies India dataset. The primary goal of this project is to predict the rating of a movie using different movie attributes such as genre, director, actors, duration, and votes.

Table of Contents
Project Description
Technologies Used
Dataset
Steps Involved
Model Evaluation
Running the Code

Project Description
This project processes and analyzes the IMDb Movies India dataset to predict movie ratings using a Random Forest Regressor model. The dataset includes various columns, such as the movie's genre, director, actors, votes, and other features. By training a machine learning model on this data, we aim to predict the rating of a movie based on the input features.

Key features in the dataset are:
Genre
Director
Actors
Duration
Votes
Year of release

Technologies Used
Python: Programming language used for data analysis and model building.
pandas: Data manipulation and analysis library.
NumPy: Library for numerical operations.
Matplotlib & Seaborn: Data visualization libraries.
Scikit-learn: Library for machine learning, used for splitting the dataset, encoding categorical variables, and training the Random Forest Regressor model.

Dataset
The dataset used in this project is named IMDb Movies India.csv, which contains information about various movies such as their ratings, genre, director, actors, and other relevant details. The dataset has been preprocessed to handle missing values and to encode categorical features.

Key Columns in the Dataset:
Name: Name of the movie
Year: Year of movie release
Genre: Genre of the movie
Director: Director of the movie
Actor 1, Actor 2, Actor 3: Main actors in the movie
Votes: Number of votes received for the movie
Duration: Duration of the movie in minutes
Rating: Rating of the movie (target variable)

Steps Involved
Data Loading and Exploration:
The dataset is loaded using pandas.
Basic data exploration is performed to examine the first few rows, column names, missing values, and summary statistics.

Data Preprocessing:
Missing values are handled by filling them with the median for numerical columns and 'Unknown' for categorical columns.
The 'Year' column is cleaned to extract only the year, and all numeric columns like 'Votes' and 'Duration' are converted to numeric values.

Feature Engineering: Additional features are created, such as:
Director_Success: The average rating of movies by each director.
Genre_Avg_Rating: The average rating of movies in each genre.

Label Encoding: Categorical columns such as Genre, Director, and Actors are label-encoded using Scikit-learn's LabelEncoder.

Model Building:
The dataset is split into training and testing sets.
A Random Forest Regressor model is trained on the training set to predict movie ratings.

Model Evaluation: The model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.
Model Evaluation
The model is evaluated using the following metrics:
Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.
Root Mean Squared Error (RMSE): Measures the square root of the average squared differences between the predicted and actual values.
R² Score: Represents the proportion of the variance in the target variable that is explained by the model.

Here is an example of the evaluation results:
Random Forest Mean Absolute Error (MAE): 0.33
Random Forest Root Mean Squared Error (RMSE): 0.61
Random Forest R² Score: 0.62

Clone the repository or download the dataset and save it as IMDb Movies India.csv in your working directory.

Install the required Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn
Run the Python script to perform data preprocessing, train the model, and evaluate its performance.
python movie_rating_prediction.py

Conclusion
This project demonstrates how to preprocess a dataset, build a machine learning model, and evaluate its performance. The Random Forest Regressor model can be improved by fine-tuning hyperparameters or using more sophisticated models such as XGBoost or neural networks.
