# Data_Science_Project-See-III-24AD084-

Flight Delay Prediction

Overview

The Flight Delay Prediction project is a data science and machine learning application designed to predict flight delays based on historical flight data. The system analyzes multiple factors such as scheduled departure times, distance, taxi times, and airline performance to estimate the likelihood of a delay. This project demonstrates an end-to-end machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, and deployment through an interactive web application built with Streamlit.

⸻

Objectives
	•	Develop a predictive model to identify potential flight delays.
	•	Analyze flight patterns and identify major factors contributing to delays.
	•	Visualize delay trends across airports, time of day, and seasons.
	•	Provide an easy-to-use interface for real-time prediction and analysis.

⸻

Key Features
	•	Data Preprocessing: Cleans, normalizes, and encodes flight data for model readiness.
	•	Feature Engineering: Includes time-based, distance, and operational features to improve model accuracy.
	•	Machine Learning Model: Implements XGBoost, optimized using GridSearchCV for high precision and recall.
	•	Imbalanced Data Handling: Uses SMOTE to balance delayed vs. on-time flight samples.
	•	Performance Metrics: Evaluates accuracy, precision, recall, and F1-score for robust model validation.
	•	Visualization: Displays delay distributions, congestion heatmaps, and temporal delay trends.
	•	Web Application: Built using Streamlit, allowing interactive single-flight and batch predictions.

⸻
Project Structure:
flight_delay_project/
│
├── app/
│   └── streamlit_app.py          # Streamlit web interface for prediction and visualization
│
├── data/
│   ├── flights.csv               # Raw dataset (from Kaggle)
│   └── new_flights.csv           # Sample test data
│
├── models/
│   ├── flight_delay_model.pkl    # Trained XGBoost model
│   └── scaler.pkl                # Scaler for preprocessing
│
├── scripts/
│   └── train_model.py            # Script for model training and evaluation
│
├── tf-venv/                      # Virtual environment directory
│
└── README.md                     # Project documentation

Dataset Description

The dataset, sourced from Kaggle’s Airline Flight Delay and Cancellation Data, includes fields such as:
	•	FL_DATE – Flight date
	•	AIRLINE – Airline code
	•	ORIGIN / DEST – Origin and destination airport codes
	•	CRS_DEP_TIME – Scheduled departure time (HHMM format)
	•	DEP_DELAY / ARR_DELAY – Departure and arrival delays (in minutes)
	•	DISTANCE – Distance between airports (in miles)
	•	TAXI_OUT / TAXI_IN – Taxi times for departure and arrival

The target variable is ARR_DELAY, which is used to determine whether a flight is Delayed (1) or On Time (0).

⸻

Model Workflow
	1.	Data Preprocessing
	•	Handling missing values using median imputation.
	•	Encoding categorical variables.
	•	Normalizing numerical features with StandardScaler.
	2.	Feature Engineering
	•	Derived features such as hour of departure, flight duration, and airport congestion indicators.
	3.	Model Training
	•	Trained using XGBoostClassifier.
	•	Hyperparameters optimized via GridSearchCV.
	•	Balanced classes using SMOTE.
	4.	Evaluation
	•	Achieved ~70% accuracy.
	•	Evaluated using precision, recall, F1-score, and confusion matrix.
	5.	Deployment
	•	Deployed via a Streamlit web interface for real-time predictions.
	•	Supports both single flight and batch CSV upload for prediction.

  Running the Project:
  1. Setup Environment:
     python3 -m venv tf-venv
source tf-venv/bin/activate   # For macOS/Linux
# or
tf-venv\Scripts\activate      # For Windows
2. Install Dependencies:
   pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit folium streamlit-folium matplotlib seaborn joblib
3. Train the Model
   python3 scripts/train_model.py
4. Run the Streamlit App
   streamlit run app/streamlit_app.py


   
Visualizations
	•	Delay Distribution: Histogram showing frequency of flight delays.
	•	Airport Congestion Map: Displays average departure delay across major airports using Folium.
	•	Daily Average Delays: Line chart showing delay trends by date.

⸻

Future Enhancements
	•	Integration with real-time weather APIs for weather-aware predictions.
	•	Deployment using Docker or AWS Cloud.
	•	Inclusion of natural language reports summarizing trends automatically.
	•	Extending support to international flight datasets.   
  
References
	1.	Kaggle – Airline Flight Delay and Cancellation Data
	2.	Scikit-learn Documentation – Model Evaluation and Preprocessing
	3.	XGBoost Official Documentation
	4.	Streamlit Developer Guide
	5.	Research Paper: Predictive Modeling of Flight Delays Using Machine Learning Techniques, IEEE (2022)

