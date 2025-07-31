import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Load model and features
clf_model = joblib.load(r'C:\Users\LOQ\OneDrive\Desktop\Heart_Disease_Project\models\final_model.pkl')

with open(r'C:\Users\LOQ\OneDrive\Desktop\Heart_Disease_Project\processed\feature_selection.json', 'r') as f:
    feature_names = json.load(f)

# Load dataset
data = pd.read_csv(r'C:\Users\LOQ\OneDrive\Desktop\Heart_Disease_Project\processed\x_features.csv')
target = pd.read_csv(r'C:\Users\LOQ\OneDrive\Desktop\Heart_Disease_Project\processed\target.csv')
data['Target'] = target


feature_labels = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "slope": "Slope of ST Segment",
    "ca": "Number of Major Vessels Colored by Fluoroscopy",

}


# --- Sidebar: User Input ---
st.sidebar.header('User Input Features')

def user_input():
    inputs = {}
    for col in feature_names:
        label = feature_labels.get(col, col)

        if col == "sex":
            gender = st.sidebar.radio(label, ("Female", "Male"))
            inputs[col] = 1 if gender == "Male" else 0

        elif data[col].dtype in ['float64', 'int64']:
            inputs[col] = st.sidebar.slider(
                label=label,
                min_value=float(data[col].min()),
                max_value=float(data[col].max()),
                value=float(data[col].mean())
            )

        else:
            inputs[col] = st.sidebar.selectbox(label, sorted(data[col].unique()))

    return pd.DataFrame([inputs])


user_data = user_input()

# --- Prediction Output ---
st.header("Prediction Result")

prediction = clf_model.predict(user_data)[0]
label = " Heart Disease Detected" if prediction == 1 else "No Heart Disease"
st.subheader(f"**{label}**")

# If model supports predict_proba
if hasattr(clf_model, "predict_proba"):
    prob = clf_model.predict_proba(user_data)[0][1]
    st.metric(label="Probability of Heart Disease", value=f"{prob:.2%}")



    # Visualization area
    st.markdown("---")
    import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("## 🧠 Heart Disease Insights Dashboard")
st.write("Select a feature to explore how it influences heart disease risk.")

# Define the plot options
plot_option = st.selectbox(
    "📌 Choose a visualization:",
    [
        "Correlation Heatmap",
        "Age Distribution",
        "Cholesterol by Target",
        "Chest Pain Type vs Target",
        "Max Heart Rate vs Age",
        "Resting ECG vs Target",
        "ST Depression (Oldpeak) by Target"
    ]
)

# Layout: 2 columns
left_col, right_col = st.columns([1.5, 3])

# Plot description and mini-metrics
with left_col:
    st.markdown("### ℹ️ Description")
    descriptions = {
        "Correlation Heatmap": "Shows how strongly each feature correlates with others including the target.",
        "Age Distribution": "Age histogram showing distribution for those with and without heart disease.",
        "Cholesterol by Target": "KDE plot for cholesterol levels by target status.",
        "Chest Pain Type vs Target": "Counts of chest pain types by presence of heart disease.",
        "Max Heart Rate vs Age": "Scatter of age vs max heart rate, colored by disease.",
        "Resting ECG vs Target": "Distribution of ECG results across disease status.",
        "ST Depression (Oldpeak) by Target": "Boxplot of ST depression levels by heart disease."
    }
    st.write(descriptions.get(plot_option, "No description available."))

    with st.expander("🔍 What is Target?"):
        st.markdown("""
        - **Target = 1** → Heart disease **present**
        - **Target = 0** → Heart disease **absent**
        """)

# Plotting section
with right_col:
    st.markdown(f"### 📊 {plot_option}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")

    if plot_option == "Correlation Heatmap":
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)

    elif plot_option == "Age Distribution":
        sns.histplot(data=data, x='age', hue='Target', kde=True, ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")

    elif plot_option == "Cholesterol by Target":
        sns.kdeplot(data=data, x='chol', hue='Target', fill=True, common_norm=False, ax=ax)
        ax.set_xlabel("Cholesterol (mg/dL)")

    elif plot_option == "Chest Pain Type vs Target":
        sns.countplot(data=data, x='cp', hue='Target', ax=ax)
        ax.set_xlabel("Chest Pain Type")

    elif plot_option == "Max Heart Rate vs Age":
        sns.scatterplot(data=data, x='age', y='thalach', hue='Target', style='Target', ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Max Heart Rate")

    elif plot_option == "Resting ECG vs Target":
        sns.countplot(data=data, x='restecg', hue='Target', ax=ax)
        ax.set_xlabel("Resting ECG Result")

    elif plot_option == "ST Depression (Oldpeak) by Target":
        sns.boxplot(data=data, x='Target', y='oldpeak', ax=ax)
        ax.set_ylabel("ST Depression (Oldpeak)")

    st.pyplot(fig)
