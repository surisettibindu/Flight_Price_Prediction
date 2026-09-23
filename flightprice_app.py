"""
✈️ Flight Price Dataset

A structured Flight Price Dataset containing 100 synthetic flight records and 15 features. This dataset is designed for data analysis, exploratory data analysis (EDA), data visualization, machine learning, and flight fare prediction projects.


📌 Project Overview

Flight ticket prices depend on several factors such as airline, travel class, route, distance, number of stops, booking time, season, and baggage allowance.

This dataset provides a compact and beginner-friendly collection of flight-related features that can be used to explore relationships between these variables and ticket prices.

> **Note:** This is a synthetic dataset created for educational and demonstration purposes. The prices do not represent real-time airline fares.


📊 Dataset Summary

| Property        |                               Value |
| --------------- | ----------------------------------: |
| File            | `flight_price_dataset_100_rows.csv` |
| Records         |                                 100 |
| Features        |                                  15 |
| Target Variable |                         `Price_INR` |
| Currency        |                             INR (₹) |
| Data Type       |                           Synthetic |
| Format          |                                 CSV |


🗂️ Dataset Features

|  # | Column                 | Data Type   | Description                           |
| -: | ---------------------- | ----------- | ------------------------------------- |
|  1 | `Flight_ID`            | String      | Unique flight identifier              |
|  2 | `Airline`              | Categorical | Airline operating the flight          |
|  3 | `Source`               | Categorical | Departure city                        |
|  4 | `Destination`          | Categorical | Arrival city                          |
|  5 | `Departure_Date`       | Date        | Scheduled departure date              |
|  6 | `Departure_Time`       | Time        | Scheduled departure time              |
|  7 | `Arrival_Time`         | Time        | Scheduled arrival time                |
|  8 | `Duration_Minutes`     | Integer     | Flight duration in minutes            |
|  9 | `Stops`                | Integer     | Number of stops                       |
| 10 | `Class`                | Categorical | Economy, Premium Economy, or Business |
| 11 | `Distance_KM`          | Integer     | Approximate flight distance           |
| 12 | `Booking_Advance_Days` | Integer     | Days between booking and departure    |
| 13 | `Season`               | Categorical | Regular, Peak, or Off-Peak            |
| 14 | `Baggage_KG`           | Integer     | Included baggage allowance            |
| 15 | `Price_INR`            | Integer     | Ticket price in Indian Rupees         |


✈️ Airlines

The dataset contains records from:

* IndiGo
* Air India
* Vistara
* SpiceJet
* Go First


🌍 Routes

Sample routes represented in the dataset include:

* Delhi → Mumbai
* Mumbai → Delhi
* Delhi → Bangalore
* Bangalore → Delhi
* Mumbai → Bangalore
* Hyderabad → Delhi
* Delhi → Hyderabad
* Chennai → Delhi
* Delhi → Kolkata
* Kolkata → Delhi
* Hyderabad → Mumbai
* Mumbai → Chennai


💺 Travel Classes

The dataset includes:

* **Economy**
* **Premium Economy**
* **Business**


🎯 Target Variable

The primary target variable is:

text
Price_INR


It represents the simulated ticket price in Indian Rupees.

This column can be used as the target for a **regression machine-learning problem**.


🔍 Possible Analysis

This dataset can be used to investigate questions such as:

* How does flight distance affect ticket price?
* Does the number of stops influence the fare?
* How does travel class affect ticket price?
* Does booking earlier result in different prices?
* How do ticket prices vary between airlines?
* How does season affect flight prices?
* Which routes have higher average fares?
* Is flight duration related to ticket price?
* Does baggage allowance have a relationship with price?


🤖 Machine Learning Applications

The dataset can be used to build a **Flight Price Prediction Model**.

Possible Input Features

text
Airline
Source
Destination
Departure_Date
Departure_Time
Duration_Minutes
Stops
Class
Distance_KM
Booking_Advance_Days
Season
Baggage_KG


Prediction Target

text
Price_INR


Possible Algorithms

You can experiment with:

* Linear Regression
* Multiple Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting
* XGBoost
* Random Forest


🧠 Example Machine Learning Workflow

Load Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Price Prediction
     ↓
Model Evaluation


📁 Project Structure

A project using this dataset could be organized as:

flight-price-prediction/
│
├── data/
│   └── flight_price_dataset_100_rows.csv
│
├── notebooks/
│   └── flight_price_analysis.ipynb
│
├── src/
│   └── model.py
│
├── README.md
│
└── requirements.txt


🎓 Suitable For

This dataset is suitable for:

* Beginner Python projects
* Pandas practice
* SQL practice
* Exploratory Data Analysis
* Data visualization
* Machine learning assignments
* Regression projects
* College projects
* Portfolio projects
* Data science practice
* Feature engineering exercises


🚀 Future Improvements

Possible extensions include:

* Add thousands of flight records.
* Include real historical flight prices.
* Add departure and arrival airports.
* Add aircraft type.
* Add weekday/weekend information.
* Add holiday indicators.
* Add days until departure.
* Add cancellation/refund information.
* Add seat availability.
* Add dynamic pricing information.
* Build a web application for price prediction.


📜 License

This dataset is provided for **educational and demonstration purposes**.

You may use, modify, and extend the dataset for learning and personal projects.


👤 Author

**Flight Price Dataset Project**

Created as a synthetic dataset for data analysis and machine-learning practice.


⭐ If You Use This Dataset

If this dataset is useful for your project, consider adding your analysis, visualizations, or machine-learning model to your project repository.

**Happy Data Analyzing! ✈️📊**  """