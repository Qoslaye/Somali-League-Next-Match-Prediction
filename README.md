# 📌 Somali Football Match Prediction

## 📖 Project Overview
This project applies **Machine Learning** to predict the outcome of Somali football matches using historical data. The model is trained to classify matches as **Win, Loss, or Draw** based on the home and away teams.

## 🛠️ Technologies Used
- **Python** (Data processing, training)
- **Pandas** (Data manipulation)
- **Matplotlib & Seaborn** (Visualizations)
- **Scikit-Learn** (Machine Learning models)
- **SMOTE** (Handling imbalanced classes)
- **Jupyter Notebook** (Development environment)


## 📝 Data Description
The dataset consists of historical Somali football match results.

| Feature | Description |
|---------|------------|
| `date` | Date of the match |
| `home` | Home team name |
| `score` | Match result (`home_goals : away_goals`) |
| `away` | Away team name |

- **Target Variable:** The match result (`Win`, `Loss`, `Draw`).
- **Data Cleaning:** Removed duplicates, handled missing values, formatted scores.

## 📊 Data Exploration & Visualization
- **Distribution of match scores** was analyzed using histograms.
- **Correlation between home and away scores** was visualized using a heatmap.

## ⚙️ Model Training & Evaluation
- **Training Algorithm:** Logistic Regression
- **Class Balancing:** Applied **SMOTE** to improve class distribution.
- **Accuracy Achieved:** ~40% (with improvements possible using better models).
- **Evaluation Metrics:** Used **confusion matrix** and **F1-score visualization**.

## 🚀 How to Run This Project
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/Qoslaye/Somali-Football-Match-Prediction.git
cd Somali-Football-Match-Prediction


