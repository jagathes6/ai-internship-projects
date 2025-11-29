import streamlit as st
import base64
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Add Background Image
# -----------------------------
def add_bg_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg_local("C:\\Users\\jagat\\Downloads\\Gemini_Generated_Image_kadcejkadcejkadc.png")
# ---- Full screen background including top navbar ----
with open("C:\\Users\\jagat\\Downloads\\Gemini_Generated_Image_kadcejkadcejkadc.png", "rb") as f:
    bg_data = f.read()
bg_encoded = base64.b64encode(bg_data).decode()

st.markdown(
    f"""
    <style>
    html, body, #root, .stApp, .main, .block-container {{
        background-image: url("data:image/png;base64,{bg_encoded}");
        background-size: cover !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Custom CSS Theme (WHITE TEXT FOR DARK BACKGROUND)
# -----------------------------
st.set_page_config(page_title="Student Grade Predictor", layout="wide")
st.markdown(
    """
    <style>

    /* Universal text readability for both dark & light backgrounds */
    body, .stApp {
        color: #ffffff;
        text-shadow: 0px 0px 4px rgba(0,0,0,0.85); /* makes white text readable on light bg */
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        color: #ffffff !important;
        text-shadow: 0px 0px 4px rgba(0,0,0,0.85);
        font-size:16px;
        font-weight: bold;
    }

    /* Buttons */
    .stButton>button {
        background-color:#4CAF50;
        color:#ffffff !important;
        font-weight:bold;
        height: 45px;
        width: 200px;
        border-radius:10px;
        font-size:16px;
        text-shadow: none;
    }

    /* Sliders & Inputs */
    label, .stSlider, .stSelectbox, .stRadio label {
        color:#ffffff !important;
        text-shadow: 0px 0px 4px rgba(0,0,0,0.85);
        font-weight:bold;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color:#ffffff !important;
        text-shadow: 0px 0px 5px rgba(0,0,0,0.9);
        font-weight:bold;
        font-family: 'Arial Black', Gadget, sans-serif;
    }

    /* Markdown text */
    .stMarkdown, p {
        color:#ffffff !important;
        text-shadow: 0px 0px 4px rgba(0,0,0,0.85);
    }

    </style>
    """,
    unsafe_allow_html=True
)



# -----------------------------
# Sidebar for Page Navigation
# -----------------------------
page = st.sidebar.radio("📌 Menu", ["Predict Grade", "Data Insights"])

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/student_data.csv")

# -----------------------------
# Predict Grade Page
# -----------------------------
if page == "Predict Grade":
    st.header("🎓 Student Grade Predictor")
    st.write("Enter student details below to predict the grade:")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Study Hours per Week", 0, 20, 5)
        attendance = st.slider("Attendance (%)", 50, 100, 80)
    with col2:
        parent_edu = st.selectbox("Parent Education", ["High School", "Bachelor", "Master"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Predict 🎯"):
        with open('model/student_model.pkl', 'rb') as f:
            model = pickle.load(f)

        input_df = pd.DataFrame({
            'study_hours':[study_hours],
            'attendance':[attendance],
            'parent_edu':[{'High School':0,'Bachelor':1,'Master':2}[parent_edu]],
            'gender':[{'Male':0,'Female':1}[gender]]
        })

        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Grade: {prediction}")

# -----------------------------
# Data Insights Page
# -----------------------------
elif page == "Data Insights":
    st.header("📊 Student Dataset Insights")

    st.subheader("Grade Distribution")
    st.bar_chart(data['grade'].value_counts())

    st.subheader("Average Attendance by Grade")
    st.bar_chart(data.groupby('grade')['attendance'].mean())

    st.subheader("Average Study Hours by Grade")
    st.bar_chart(data.groupby('grade')['study_hours'].mean())

    st.subheader("Feature Importance")
    with open('model/student_model.pkl', 'rb') as f:
        model = pickle.load(f)
    X = data[['study_hours','attendance','parent_edu','gender']]
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax)
    st.pyplot(fig)
