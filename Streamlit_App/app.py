import streamlit as st
import pandas as pd
import joblib
import time
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "Dataset" / "boston.csv"
MODEL_PATH = BASE_DIR / "Model" / "house_price_model.pkl"
FEATURE_PATH = BASE_DIR / "Model" / "features.pkl"

df = pd.read_csv(DATASET_PATH)
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)
# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="House Price Prediction",
    
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------------------
# LOAD CSS
# ---------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ---------------------------
# SIDEBAR
# ---------------------------
LOGO_PATH = BASE_DIR / "Images" / "logo.png"
st.sidebar.image(str(LOGO_PATH), width=180)
st.sidebar.title("House Price Prediction")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.info("""
### Machine Learning

✅ Random Forest Regressor

Accuracy (R²)

**89.25%**
""")

st.sidebar.markdown("---")

st.sidebar.write("⚡ Tech Stack")

st.sidebar.write("🐍 Python")
st.sidebar.write("🤖 Scikit-Learn")
st.sidebar.write("🌐 Streamlit")
st.sidebar.write("📊 Pandas")
st.sidebar.write("📈 Plotly")






# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div class="hero-card">

<div class="main-title">
🏠 House Price Prediction Dashboard
</div>

<div class="sub-title">
Predict House Prices using Artificial Intelligence and Machine Learning.
</div>

<br>

✅ Random Forest Model &nbsp;&nbsp;&nbsp;
📊 Accuracy: 89.25% &nbsp;&nbsp;&nbsp;

</div>
""", unsafe_allow_html=True)
st.divider()

# ---------------------------
# METRICS
# ---------------------------
c1,c2,c3,c4=st.columns(4)

cards=[
("Accuracy","89.25%"),
("Models","3"),
("Features","13"),
("Dataset","506")
]

for col,(title,value) in zip([c1,c2,c3,c4],cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-title">{title}</div>

        <div class="metric-value">{value}</div>

        </div>
        """,unsafe_allow_html=True)

st.divider()

# ---------------------------
# INPUTS
# ---------------------------

st.subheader("Enter House Details")

col1,col2 = st.columns(2)

inputs={}

with col1:

    inputs["CRIM"]=st.number_input("Crime Rate",0.0)

    inputs["ZN"]=st.number_input("Residential Land",0.0)

    inputs["INDUS"]=st.number_input("Industrial Area",0.0)

    inputs["CHAS"]=st.number_input("Charles River",0,1)

    inputs["NOX"]=st.number_input("Nitric Oxide",0.0)

    inputs["RM"]=st.number_input("Average Rooms",0.0)

    inputs["AGE"]=st.number_input("House Age",0.0)

with col2:

    inputs["DIS"]=st.number_input("Distance",0.0)

    inputs["RAD"]=st.number_input("Highway Access",0)

    inputs["TAX"]=st.number_input("Property Tax",0)

    inputs["PTRATIO"]=st.number_input("Pupil Teacher Ratio",0.0)

    inputs["B"]=st.number_input("B",0.0)

    inputs["LSTAT"]=st.number_input("Lower Status Population",0.0)

st.write("")

# ---------------------------
# BUTTON
# ---------------------------
if st.button("Predict House Price", use_container_width=True):

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    with st.spinner("Predicting..."):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)

    # Keep this INSIDE the if block
    st.markdown(f"""
    <div class="prediction-card">
        <h2>🏠 Estimated House Price</h2>
        <h1>₹ {prediction[0]:,.2f} Lakhs</h1>
        <p>Prediction completed successfully.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.divider()
st.markdown("""
<h2 style='color:#1E3A8A'>
📊 Data Visualization
</h2>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📈 Price Distribution",
    "📉 Feature Correlation",
    "📊 Average Price by Rooms"
])

with tab1:
    fig = px.histogram(
        df,
        x="MEDV",
        nbins=30,
        title="Distribution of House Prices"
    )

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Correlation Heatmap"
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    room_df = df.groupby("RM")["MEDV"].mean().reset_index()

    fig = px.line(
        room_df,
        x="RM",
        y="MEDV",
        markers=True,
        title="Average House Price vs Number of Rooms"
    )

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)# ---------------------------
# ABOUT
# ---------------------------
st.markdown("""
<hr>

<center>

🏠 <b>House Price Prediction System</b><br>
AIML Summer Internship 2026

</center>
""", unsafe_allow_html=True)
