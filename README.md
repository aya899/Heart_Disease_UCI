# Heart Disease Prediction with Machine Learning

This project predicts the presence of heart disease using machine learning models trained on clinical data. The app is deployed using [Streamlit](https://streamlit.io) and allows users to interactively test predictions and explore visual insights.


Live App: [https://app-deploy-889.streamlit.app/]
GitHub Repo: [https://github.com/aya899/Heart_Disease_UCI]


# Project Structure

Heart_Disease_UCI/
│
├── data/ #dataset
├── processed/ # Cleaned data, selected features, PCA, x_features, target
├── models/ #final_model.pkl
├── notebooks/ # Jupyter notebooks for preprocessing, modeling
├── ui/app.py # Streamlit app file
├── requirements.txt # App dependencies
└── README.md # This file


# Features Used

- `sex` – Gender
- `cp` – Chest pain type
- `exang` – Exercise-induced angina
- `oldpeak` – ST depression
- `restecg` – Resting ECG results
- `slope` – Slope of ST segment
- `thal` – Thalassemia result
- `ca` – Number of vessels colored by fluoroscopy


# Machine Learning Pipeline

- **Feature Selection**: Chi-Square, RFE, Random Forest
- **Models Used**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score
  - ROC Curve & AUC


# Streamlit Dashboard

Includes:
- Prediction based on user input
- Probability estimate of heart disease
- Interactive visualizations:
  - Chest Pain Type vs Target
  - Number of Vessels vs Target
  - Slope of ST Segment vs Target
  - ST Depression vs Target
  - Sex & Thalassemia analysis
  - Correlation Heatmap
  - Exercise-Induced Angina vs Target
  - Resting ECG vs Target


# Tools Used

- Python 3.11
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn


# Author

**Aya Ramadan**  

[GitHub](https://github.com/aya899)  
[Linkedin](https://www.linkedin.com/in/aya-ramadan-907545329/)

