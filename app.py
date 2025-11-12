import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ===========================
# Load Models & Dataset
# ===========================
emi_eligibility_model = joblib.load("C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/best_model_EMI_Eligibility_Prediction.pkl")
max_emi_model = joblib.load("C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/best_regressor_model_max_emi _amount.pkl")

# Load dataset for visualization
df_data = pd.read_csv("C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/final_dataset.csv")

# ===========================
# Streamlit Config
# ===========================
st.set_page_config(page_title="EMI Prediction Dashboard", layout="wide")
st.title("💰 EMIPredict AI - Intelligent Financial Risk Assessment Platform")

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function
add_bg_from_local("C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/emi_prediction/bg.png")

# ===========================
# Preprocessing Function
# ===========================
def preprocess_input(df):
    mappings = {
        'gender': {'Male': 0, 'Female': 1},
        'marital_status': {'Single': 0, 'Married': 1},
        'education': {'High School': 0, 'Graduate': 1, 'Postgraduate': 2, 'PhD': 3},
        'employment_type': {'Salaried': 0, 'Self-Employed': 1},
        'company_type': {'Private': 0, 'Government': 1, 'Startup': 2, 'Other': 3},
        'house_type': {'Owned': 0, 'Rented': 1},
        'emi_scenario': {'New': 0, 'Upgrade': 1, 'Top-up': 2},
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = df.fillna(0)
    return df

# ===========================
# User Input Section
# ===========================
def get_user_input():
    st.subheader("📋 Enter Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 75, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education Level", ["High School", "Graduate", "Postgraduate", "PhD"])
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
        company_type = st.selectbox("Company Type", ["Private", "Government", "Startup", "Other"])
        years_of_employment = st.number_input("Years of Employment", 0, 50, 5)
        house_type = st.selectbox("House Type", ["Owned", "Rented"])
        family_size = st.number_input("Family Size", 1, 15, 4)
        dependents = st.number_input("Dependents", 0, 10, 2)

    with col2:
        monthly_salary = st.number_input("Monthly Salary (₹)", 0, step=1000, value=50000)
        monthly_rent = st.number_input("Monthly Rent (₹)", 0, step=500, value=10000)
        existing_loans = st.number_input("Existing Loans (₹)", 0, step=500, value=0)
        current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, step=500, value=0)
        bank_balance = st.number_input("Bank Balance (₹)", 0, step=1000, value=20000)
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0, step=1000, value=10000)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, step=500, value=8000)
        travel_expenses = st.number_input("Travel Expenses (₹)", 0, step=500, value=2000)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, step=500, value=1000)
        school_fees = st.number_input("School Fees (₹)", 0, step=500, value=0)
        college_fees = st.number_input("College Fees (₹)", 0, step=500, value=0)

    st.subheader("🏠 Loan Details")
    requested_amount = st.number_input("Requested Loan Amount (₹)", 1000, step=1000, value=100000)
    requested_tenure = st.number_input("Requested Tenure (months)", 6, 120, 24)
    emi_scenario = st.selectbox("EMI Scenario", ["New", "Upgrade", "Top-up"])

    return {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }

# ===========================
# EMI Eligibility Page
# ===========================
if st.session_state.page == "emi_eligibility":
    st.header("🔍 EMI Eligibility Prediction")

    user_input = get_user_input()
    max_monthly_emi = st.number_input("Predicted Max Monthly EMI (₹)", 0, step=1000, value=25000)
    user_input["max_monthly_emi"] = max_monthly_emi
    df_user = pd.DataFrame([user_input])
    processed_df = preprocess_input(df_user)

    if st.button("Predict EMI Eligibility"):
        result = emi_eligibility_model.predict(processed_df)[0]
        if result == 1:
            st.success("✅ You are eligible for EMI!")
        else:
            st.error("❌ Not eligible for EMI.")

        # Visualization Section
        st.subheader("📊 Your Data vs Dataset Trends")

        fig1 = px.scatter(df_data, x="monthly_salary", y="max_monthly_emi", color="emi_eligibility",
                          title="Monthly Salary vs Max EMI (Eligibility Highlight)")
        fig1.add_scatter(x=[user_input["monthly_salary"]], y=[max_monthly_emi],
                         mode="markers+text", text=["You"], marker=dict(size=14, color="red"), name="Your Data")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df_data, x="credit_score", nbins=30, title="Credit Score Distribution")
        fig2.add_vline(x=user_input["credit_score"], line_color="red", annotation_text="You", line_width=3)
        st.plotly_chart(fig2, use_container_width=True)

        # --- Descriptions for EMI Eligibility Graphs ---
        st.markdown("""
        **📘 Scatter Plot Description:**  
        This plot shows how **Monthly Salary** relates to **Max EMI**.  
        Each point represents a user from the dataset, color-coded by eligibility.  
        The red dot is your input — showing where you stand in relation to others.

        **📘 Distribution Plot Description:**  
        This histogram shows how credit scores are distributed among applicants.  
        The red vertical line marks **your credit score**, so you can see how you compare to others.
        """)

    st.info("💡 Want to know your Max EMI Amount?")
    st.button("Go to Max EMI Calculator", on_click=lambda: go_to("max_emi"))

# ===========================
# Max EMI Page
# ===========================
elif st.session_state.page == "max_emi":
    st.header("📈 Max EMI Amount Prediction")

    user_input = get_user_input()
    emi_eligibility = st.selectbox("Are you EMI Eligible?", [0, 1])
    user_input["emi_eligibility"] = emi_eligibility
    df_user = pd.DataFrame([user_input])
    processed_df = preprocess_input(df_user)

    if st.button("Predict Max EMI Amount"):
        result = max_emi_model.predict(processed_df)[0]
        st.markdown(f"<h3 style='color:black;'>💰 Your maximum affordable EMI is ₹{result:.2f}</h3>", unsafe_allow_html=True)


        # Visualization Section
        st.subheader("📊 Your Data vs Dataset Trends")

        fig1 = px.scatter(df_data, x="monthly_salary", y="max_monthly_emi", color="emi_eligibility",
                          title="Monthly Salary vs Max EMI by Eligibility")
        fig1.add_scatter(x=[user_input["monthly_salary"]], y=[result],
                         mode="markers+text", text=["You"], marker=dict(size=14, color="red"), name="Your Data")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df_data, x="bank_balance", nbins=30, title="Bank Balance Distribution")
        fig2.add_vline(x=user_input["bank_balance"], line_color="red", annotation_text="You", line_width=3)
        st.plotly_chart(fig2, use_container_width=True)

        # --- Descriptions for Max EMI Graphs ---
        st.markdown("""
        **📘 Scatter Plot Description:**  
        This scatter plot shows the relationship between **Monthly Salary** and **Max EMI**,  
        helping you visualize how your salary influences your possible EMI.  
        The red dot represents your data point for easy comparison.

        **📘 Distribution Plot Description:**  
        This histogram displays the distribution of **Bank Balances** across applicants.  
        The red vertical line indicates **your bank balance**, allowing you to see where you stand among others.
        """)

    st.info("💡 Want to check your EMI eligibility?")
    st.button("Go to EMI Eligibility Checker", on_click=lambda: go_to("emi_eligibility"))

# ===========================
# Home Page
# ===========================
else:
    st.subheader("Choose your prediction 👇")
    col1, col2 = st.columns(2)
    with col1:
        st.button("🔍 Check EMI Eligibility", on_click=lambda: go_to("emi_eligibility"))
    with col2:
        st.button("📈 Calculate Max EMI Amount", on_click=lambda: go_to("max_emi"))

    st.markdown("---")
    st.caption("⚙️ Powered by trained ML models + Visualization Dashboard")
