import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Configure page
st.set_page_config(
    page_title="PharmaDemand ERP Integration",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .header-style {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        background-color: #2b5876;
        margin-bottom: 15px;
    }
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-card h3 {
        color: #000000;  /* Black for headers */
    }
    .metric-card h2 {
        color: #000000;  /* Black for values */
    }
    .reorder-alert {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffa000;
        margin-bottom: 15px;
    }
    .reorder-alert h3, 
    .reorder-alert p {
        color: #000000 !important;  /* Force black text */
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("processed_data/medicine_inventory_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/demand_forecasting_rf.pkl")

# Generate predictions
def generate_predictions(model, data, months_ahead=2):
    features = [
        'SKU', 'Stock In', 'Stock Out', 'Stock Balance', 'Days_Until_Expiry', 'Lead_Time',
        'Season', 'Holiday', 'Promotion', 'Lag_1', 'Lag_2', 'Lag_3', 
        'Moving Average', 'Order_Day', 'Order_Month', 'Order_Weekday', 'Days_Since_Last_Order'
    ]
    
    all_forecasts = []
    
    for month_offset in range(1, months_ahead + 1):
        forecast_date = datetime.now() + relativedelta(months=month_offset)
        forecast_month = forecast_date.month
        
        # Create prediction dataset
        pred_data = data.copy()
        pred_data['Order Month'] = forecast_month
        pred_data['Order Year'] = forecast_date.year
        
        # Make predictions
        pred_data['Predicted Demand'] = model.predict(pred_data[features])
        
        # Aggregate by SKU
        monthly_forecast = pred_data.groupby(['SKU']).agg({
            'Predicted Demand': 'sum',
            'Stock Balance': 'last',
            'Lead_Time': 'mean'
        }).reset_index()
        
        # Calculate reorder logic
        monthly_forecast['Reorder Threshold'] = monthly_forecast['Predicted Demand'] * 0.3  # 30% of predicted demand
        monthly_forecast['Reorder Needed'] = monthly_forecast['Stock Balance'] < monthly_forecast['Reorder Threshold']
        monthly_forecast['Reorder Quantity'] = np.where(
            monthly_forecast['Reorder Needed'],
            monthly_forecast['Predicted Demand'] - monthly_forecast['Stock Balance'],
            0
        )
        monthly_forecast['Recommended Order Date'] = forecast_date.replace(day=1).strftime('%Y-%m-%d')
        monthly_forecast['Month'] = forecast_date.strftime('%B %Y')
        
        all_forecasts.append(monthly_forecast)
    
    return pd.concat(all_forecasts, ignore_index=True)

# Main app
def main():
    st.title("🏥 PharmaDemand ERP Integration Dashboard")
    st.markdown("""
    **AI-powered demand forecasting solution** for pharmaceutical inventory optimization
    """)
    
    # Load data and model
    data = load_data()
    model = load_model()
    
    # Sidebar - ERP integration demo
    with st.sidebar:
        st.subheader("ERP Integration Settings")
        erp_connected = st.checkbox("Simulate ERP Connection", value=True)
        auto_ordering = st.checkbox("Enable Automated Ordering", value=True)
        alert_threshold = st.slider("Low Stock Alert Threshold (%)", 20, 50, 30)
        
        st.markdown("---")
        st.subheader("Forecast Parameters")
        forecast_months = st.slider("Months to Forecast", 1, 6, 2)
        lead_time_buffer = st.slider("Lead Time Buffer (days)", 1, 14, 7)
        
        st.markdown("---")
        st.markdown("**System Status**")
        if erp_connected:
            st.success("✅ Connected to ERP System")
        else:
            st.warning("⚠️ ERP Not Connected")
        
        if auto_ordering:
            st.info("🤖 Auto-Ordering Enabled")
    
    # Generate predictions
    predictions = generate_predictions(model, data, forecast_months)
    
    # Main dashboard
    tab1, tab2, tab3 = st.tabs(["📋 Inventory Overview", "📅 Order Recommendations", "⚙️ ERP Integration"])
    
    with tab1:
        st.markdown('<div class="header-style">Current Inventory Status</div>', unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            selected_sku = st.selectbox(
                "Select SKU",
                options=data['SKU'].unique(),
                format_func=lambda x: f"SKU-{x}"
            )
        with col2:
            view_mode = st.radio(
                "View Mode",
                ["Single SKU", "All Inventory"],
                horizontal=True
            )
        
        if view_mode == "Single SKU":
            sku_data = predictions[predictions['SKU'] == selected_sku]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">'
                           f"<h3>Current Stock</h3>"
                           f"<h2>{int(sku_data.iloc[0]['Stock Balance'])}</h2>"
                           "</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">'
                           f"<h3>Predicted Demand</h3>"
                           f"<h2>{int(sku_data.iloc[0]['Predicted Demand'])}</h2>"
                           "</div>", unsafe_allow_html=True)
            
            with col3:
                status = "🟢 Sufficient" if not sku_data.iloc[0]['Reorder Needed'] else "🔴 Low Stock"
                st.markdown('<div class="metric-card">'
                           f"<h3>Inventory Status</h3>"
                           f"<h2>{status}</h2>"
                           "</div>", unsafe_allow_html=True)
            
            # Display forecast table
            st.markdown('<div class="header-style">Monthly Forecast</div>', unsafe_allow_html=True)
            st.dataframe(
                sku_data[['Month', 'Predicted Demand', 'Stock Balance', 'Reorder Needed']]
                .rename(columns={
                    'Month': 'Forecast Month',
                    'Predicted Demand': 'Demand Forecast',
                    'Stock Balance': 'Current Stock',
                    'Reorder Needed': 'Reorder Alert'
                }),
                hide_index=True,
                use_container_width=True
            )
        
        else:
            # Display all inventory status
            st.markdown('<div class="header-style">Complete Inventory Status</div>', unsafe_allow_html=True)
            
            # Add filters for reorder status
            alert_filter = st.selectbox(
                "Filter by Alert Status",
                ["All", "Only Reorder Alerts", "No Alerts"]
            )
            
            filtered_data = predictions.copy()
            if alert_filter == "Only Reorder Alerts":
                filtered_data = filtered_data[filtered_data['Reorder Needed']]
            elif alert_filter == "No Alerts":
                filtered_data = filtered_data[~filtered_data['Reorder Needed']]
            
            st.dataframe(
                filtered_data[['SKU', 'Month', 'Predicted Demand', 'Stock Balance', 'Reorder Needed']]
                .rename(columns={
                    'SKU': 'Product SKU',
                    'Month': 'Forecast Month',
                    'Predicted Demand': 'Demand Forecast',
                    'Stock Balance': 'Current Stock',
                    'Reorder Needed': 'Reorder Alert'
                }),
                hide_index=True,
                use_container_width=True
            )
    
    with tab2:
        st.markdown('<div class="header-style">Order Recommendations</div>', unsafe_allow_html=True)
        
        # Filter for items needing reorder
        reorder_items = predictions[predictions['Reorder Needed']].copy()
        
        if not reorder_items.empty:
            # Group by month
            for month in reorder_items['Month'].unique():
                month_data = reorder_items[reorder_items['Month'] == month]
                
                st.markdown(f'<div class="reorder-alert">'
                           f'<h3>📅 {month} Reorders</h3>'
                           f'<p>{len(month_data)} products require replenishment</p>'
                           '</div>', unsafe_allow_html=True)
                
                # Display order recommendations
                for _, row in month_data.iterrows():
                    order_date = (datetime.strptime(row['Recommended Order Date'], '%Y-%m-%d') - 
                                timedelta(days=row['Lead_Time'] + lead_time_buffer))
                    
                    with st.expander(f"SKU-{row['SKU']} - {int(row['Reorder Quantity'])} units"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            **Order Details**
                            - Current Stock: {int(row['Stock Balance'])}
                            - Predicted Demand: {int(row['Predicted Demand'])}
                            - Reorder Quantity: **{int(row['Reorder Quantity'])}**
                            - Lead Time: {int(row['Lead_Time'])} days
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **Timing**
                            - Forecast Month: {month}
                            - Recommended Order Date: {order_date.strftime('%Y-%m-%d')}
                            - Expected Delivery: {row['Recommended Order Date']}
                            """)
                        
                        if auto_ordering:
                            if st.button(f"📝 Generate PO for SKU-{row['SKU']}", key=f"po_{row['SKU']}_{month}"):
                                st.success(f"""
                                **Purchase Order Generated**
                                - SKU: {row['SKU']}
                                - Quantity: {int(row['Reorder Quantity'])}
                                - Order Date: {order_date.strftime('%Y-%m-%d')}
                                - Expected Delivery: {row['Recommended Order Date']}
                                """)
        else:
            st.success("No reorder recommendations at this time")
    
    with tab3:
        st.markdown('<div class="header-style">ERP Integration Module</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="erp-integration">'
                   '<h3>Seamless ERP Integration</h3>'
                   '<p>Our solution can integrate directly with your existing ERP system to:</p>'
                   '<ul>'
                   '<li>Automatically sync inventory levels in real-time</li>'
                   '<li>Generate purchase orders when thresholds are breached</li>'
                   '<li>Update demand forecasts based on latest sales data</li>'
                   '<li>Provide dashboard visibility across all locations</li>'
                   '</ul>'
                   '</div>', unsafe_allow_html=True)
        
        # ERP integration demo
        st.subheader("Integration Demo")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Current Integration Status**")
            if erp_connected:
                st.success("✅ Successfully connected to ERP API")
                st.metric("Last Sync", datetime.now().strftime("%Y-%m-%d %H:%M"))
                st.metric("SKUs Synced", len(data))
            else:
                st.warning("ERP connection not configured")
        
        with col2:
            st.markdown("**Available ERP Systems**")
            erp_system = st.selectbox(
                "Select your ERP",
                ["SAP", "Oracle", "Microsoft Dynamics", "Infor", "Other"]
            )
            
            if st.button("Test Connection"):
                if erp_connected:
                    st.success(f"Successfully connected to {erp_system} ERP system")
                else:
                    st.error("Connection failed - check credentials")
        
        # API documentation
        with st.expander("🛠️ Integration Technical Details"):
            st.markdown("""
            **REST API Endpoints**
            ```
            GET /api/inventory          # Get current inventory levels
            POST /api/orders            # Create new purchase orders
            GET /api/forecasts          # Retrieve demand forecasts
            ```
            
            **Data Formats**
            ```json
            {
              "sku": "12345",
              "current_stock": 150,
              "predicted_demand": 420,
              "reorder_quantity": 270,
              "order_date": "2023-11-15"
            }
            ```
            """)

if __name__ == "__main__":
    main()