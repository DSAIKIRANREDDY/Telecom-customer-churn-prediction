
import streamlit as st

from components.dashboard import show_dashboard
from components.single_prediction import show_single_prediction
from components.batch_prediction import show_batch_prediction
from components.about import show_about

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Application Navigation
# --------------------------------------------------

PAGES = [
    "Dashboard",
    "Single Prediction",
    "Batch Prediction",
    "About"
]


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Telecom Churn")
st.sidebar.caption("Customer Churn Prediction System")

st.sidebar.divider()

selected_page = st.sidebar.radio(
    "Navigation",
    PAGES,
    index=0
)

st.sidebar.divider()

st.sidebar.caption("Powered by Machine Learning")


# --------------------------------------------------
# Page Routing
# --------------------------------------------------

if selected_page == "Dashboard":
    show_dashboard()

elif selected_page == "Single Prediction":
    show_single_prediction()

elif selected_page == "Batch Prediction":
    show_batch_prediction()

elif selected_page == "About":
    show_about()
