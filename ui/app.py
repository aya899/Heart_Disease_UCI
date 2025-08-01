import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

model_path = os.path.join("models", "final_model.pkl")
clf_model = joblib.load(model_path)


json_path = os.path.join("processed", "feature_selection.json")
with open(json_path, 'r') as f:
    feature_names = json.load(f) 

data = pd.read_csv(os.path.join("processed", "x_features.csv"))
target = pd.read_csv(os.path.join("processed", "target.csv"))
data['Target'] = target


feature_labels = {
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "slope": "Slope of ST Segment",
    "ca": "Number of Major Vessels Colored by Fluoroscopy",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "restecg": "Resting ECG Result",
    "thal": "Thalassemia Type"
}




st.sidebar.header('User Input Features')

def user_input():
    inputs = {}
    for col in feature_names:
        label = feature_labels.get(col, col)

        if col == "sex":
            gender = st.sidebar.radio(label, ("Female", "Male"))
            inputs[col] = 1 if gender == "Male" else 0

        elif col == "exang":
            val = st.sidebar.radio(label, ("No", "Yes"))
            inputs[col] = 1 if val == "Yes" else 0

        elif col == "cp":
            options = {
                "Typical Angina (0)": 0,
                "Atypical Angina (1)": 1,
                "Non-anginal Pain (2)": 2,
                "Asymptomatic (3)": 3
            }
            choice = st.sidebar.selectbox(label, options.keys())
            inputs[col] = options[choice]

        elif col == "slope":
            options = {
                "Upsloping (0)": 0,
                "Flat (1)": 1,
                "Downsloping (2)": 2
            }
            choice = st.sidebar.selectbox(label, options.keys())
            inputs[col] = options[choice]

        elif col == "restecg":
            options = {
                "Normal (0)": 0,
                "ST-T Abnormality (1)": 1,
                "Left Ventricular Hypertrophy (2)": 2
            }
            choice = st.sidebar.selectbox(label, options.keys())
            inputs[col] = options[choice]

        elif col == "thal":
            options = {
                "Normal (1)": 1,
                "Fixed Defect (2)": 2,
                "Reversible Defect (3)": 3
            }
            choice = st.sidebar.selectbox(label, options.keys())
            inputs[col] = options[choice]

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


#prediction phase
st.header("Prediction Result")

prediction = clf_model.predict(user_data)[0]
label = " Heart Disease Detected" if prediction == 1 else "No Heart Disease"
st.subheader(f"**{label}**")

#predict_proba
if hasattr(clf_model, "predict_proba"):
    prob = clf_model.predict_proba(user_data)[0][1]
    st.metric(label="Probability of Heart Disease", value=f"{prob:.2%}")



# Visualization
    st.markdown("---")
    

st.markdown("Heart Disease Insights Dashboard")
st.write("Select a feature to explore how it influences heart disease risk.")


plot_option = st.selectbox(
    "Choose a visualization:",
    [
        "Correlation Heatmap",
        "Chest Pain Type vs Target",
        "Number of Vessels vs Target",
        "Slope of ST Segment vs Target",
        "Exercise-Induced Angina vs Target",
        "Sex vs Target",
        "ST Depression vs Target",
        "Resting ECG vs Target",
        "Thalassemia vs Target"
    ]
)




left_col, right_col = st.columns([1.5, 3])

# Plot description
with left_col:
    st.markdown("Description")
    descriptions = {
        "Correlation Heatmap": "Displays correlation values between all selected features and the target. Helps identify which variables are most related to heart disease.",
        "Chest Pain Type vs Target": "Shows the count of different chest pain types, split by heart disease status.",
        "Number of Vessels vs Target": "Visualizes how the number of major vessels colored by fluoroscopy relates to heart disease.",
        "Slope of ST Segment vs Target": "Displays how different ST segment slopes are distributed across the heart disease classes.",
        "Exercise-Induced Angina vs Target": "Compares the presence or absence of exercise-induced angina across heart disease outcomes.",
        "Sex vs Target": "Compares heart disease distribution between male and female patients.",
        "ST Depression vs Target": "Boxplot showing the distribution of ST depression (oldpeak) values grouped by disease presence.",
        "Resting ECG vs Target": "Shows the distribution of resting ECG results by heart disease outcome.",
        "Thalassemia vs Target": "Visualizes the types of thalassemia and how they relate to the presence or absence of heart disease."
    }

    st.write(descriptions.get(plot_option, "No description available."))

    with st.expander("What is Target?"):
        st.markdown("""
        - **Target = 1** → Heart disease **present**
        - **Target = 0** → Heart disease **absent**
        """)

# Plotting section
with right_col:
    st.markdown(f"{plot_option}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")

    if plot_option == "Correlation Heatmap":
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)

    elif plot_option == "Chest Pain Type vs Target":
        cp_labels = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}
        data_plot = data.copy()
        data_plot["cp"] = data_plot["cp"].map(cp_labels)
        sns.countplot(data=data_plot, x='cp', hue='Target', ax=ax)
        ax.set_xlabel("Chest Pain Type")

    elif plot_option == "Number of Vessels vs Target":
        sns.countplot(data=data, x='ca', hue='Target', ax=ax)
        ax.set_xlabel("Number of Major Vessels")

    elif plot_option == "Slope of ST Segment vs Target":
        slope_labels = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
        data_plot = data.copy()
        data_plot["slope"] = data_plot["slope"].map(slope_labels)
        sns.countplot(data=data_plot, x='slope', hue='Target', ax=ax)
        ax.set_xlabel("Slope of ST Segment")

    elif plot_option == "Exercise-Induced Angina vs Target":
        exang_labels = {0: "No", 1: "Yes"}
        data_plot = data.copy()
        data_plot["exang"] = data_plot["exang"].map(exang_labels)
        sns.countplot(data=data_plot, x='exang', hue='Target', ax=ax)
        ax.set_xlabel("Exercise-Induced Angina")

    elif plot_option == "Sex vs Target":
        sex_labels = {0: "Female", 1: "Male"}
        data_plot = data.copy()
        data_plot["sex"] = data_plot["sex"].map(sex_labels)
        sns.countplot(data=data_plot, x='sex', hue='Target', ax=ax)
        ax.set_xlabel("Sex")

    elif plot_option == "ST Depression vs Target":
        sns.boxplot(data=data, x='Target', y='oldpeak', ax=ax)
        ax.set_ylabel("ST Depression (Oldpeak)")

    elif plot_option == "Resting ECG vs Target":
        restecg_labels = {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}
        data_plot = data.copy()
        data_plot["restecg"] = data_plot["restecg"].map(restecg_labels)
        sns.countplot(data=data_plot, x='restecg', hue='Target', ax=ax)
        ax.set_xlabel("Resting ECG")

    elif plot_option == "Thalassemia vs Target":
        thal_labels = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
        data_plot = data.copy()
        data_plot["thal"] = data_plot["thal"].map(thal_labels)
        sns.countplot(data=data_plot, x='thal', hue='Target', ax=ax)
        ax.set_xlabel("Thalassemia")

    ax.set_ylabel("Count")
    st.pyplot(fig)
