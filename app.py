import streamlit as st
import pandas as pd
import plotly.express as px

# Load forecast data
def load_data():
    file_path = "predictions/multi_month_demand_forecast.csv"
    return pd.read_csv(file_path)
# Streamlit App
st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")
st.title("📊 SKU Demand Forecast & Reorder Recommendations")

# Load Data
df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")
sku_filter = st.sidebar.text_input("Search SKU:")
reorder_filter = st.sidebar.selectbox("Show Only Reorders:", ["All", True, False])

# Apply Filters
if sku_filter:
    df = df[df['SKU'].astype(str).str.contains(sku_filter, case=False, na=False)]
if reorder_filter != "All":
    df = df[df['Reorder Needed'] == reorder_filter]

# Display Data
st.dataframe(df, use_container_width=True)

# Visualization: Demand Forecast vs Stock Balance
fig = px.line(df, x='Order Month', y=['Predicted Demand', 'Stock Balance'], color='SKU',
              title='Predicted Demand vs Stock Balance', markers=True)
st.plotly_chart(fig, use_container_width=True)

# Download Button
st.sidebar.download_button("Download Forecast CSV", df.to_csv(index=False), "forecast_data.csv", "text/csv")