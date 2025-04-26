import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
from sklearn.preprocessing import StandardScaler
import os

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Function to load the model
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Title and description
st.title("⚡ Energy Consumption Prediction")
st.markdown("""
This application predicts energy consumption based on environmental and operational parameters.
Enter the input values in the sidebar and click 'Predict' to get results.
""")

# Create sidebar for inputs
st.sidebar.header("Input Parameters")

# Load the model (update the path to where your model is located)
model_path = "C:/Users/JAYA SOORYA/Downloads/Energy prediction/energy_prediction_model.pkl"
model = load_model(model_path)

# Default values (modify these according to your actual data ranges)
default_temp = 22.0
default_humidity = 45.0
default_square_footage = 2000
default_occupancy = 10
default_renewable = 20.0

# Input fields in sidebar
with st.sidebar:
    st.subheader("Environmental Parameters")
    temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=40.0, value=default_temp, step=0.5)
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=default_humidity, step=1.0)
    
    st.subheader("Building Parameters")
    square_footage = st.number_input("Square Footage", min_value=100, max_value=10000, value=default_square_footage)
    occupancy = st.number_input("Occupancy (people)", min_value=0, max_value=500, value=default_occupancy)
    
    st.subheader("System States")
    hvac_usage = st.selectbox("HVAC Usage", options=["On", "Off"])
    lighting_usage = st.selectbox("Lighting Usage", options=["On", "Off"])
    renewable_energy = st.slider("Renewable Energy (%)", min_value=0.0, max_value=100.0, value=default_renewable, step=1.0)
    
    st.subheader("Time Parameters")
    date_selected = st.date_input("Date", datetime.datetime.now().date())
    day_of_week = date_selected.strftime("%A")
    st.info(f"Day of Week: {day_of_week}")
    
    holiday = st.selectbox("Holiday", options=["No", "Yes"])

    # Calculate derived features
    temp_humid_interaction = temperature * humidity
    occupancy_per_sqft = occupancy / square_footage if square_footage > 0 else 0
    
    # Extract date components
    selected_datetime = datetime.datetime.combine(date_selected, datetime.datetime.min.time())
    hour = 12  # Default to noon
    day = selected_datetime.day
    month = selected_datetime.month
    year = selected_datetime.year
    day_of_year = selected_datetime.timetuple().tm_yday
    week_of_year = int(selected_datetime.strftime("%V"))
    
    # Create cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)

# Main content area
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Visualization", "Model Information"])

# Tab 1: Prediction
with tab1:
    st.header("Energy Consumption Prediction")
    
    # Create a function to prepare input data for prediction
    def prepare_input_data():
        # Create a dataframe with a single row containing all the input features
        data = {
            'Temperature': [temperature],
            'Humidity': [humidity],
            'SquareFootage': [square_footage],
            'Occupancy': [occupancy],
            'HVACUsage': [hvac_usage],
            'LightingUsage': [lighting_usage],
            'RenewableEnergy': [renewable_energy],
            'DayOfWeek': [day_of_week],
            'Holiday': [holiday],
            'Hour': [hour],
            'Day': [day],
            'Month': [month],
            'Year': [year],
            'DayOfYear': [day_of_year],
            'WeekOfYear': [week_of_year],
            'Hour_sin': [hour_sin],
            'Hour_cos': [hour_cos],
            'DayOfYear_sin': [day_of_year_sin],
            'DayOfYear_cos': [day_of_year_cos],
            'Temp_Humid_Interaction': [temp_humid_interaction],
            'Occupancy_per_SqFt': [occupancy_per_sqft]
        }
        return pd.DataFrame(data)
    
    # Button to make prediction
    if st.button("Predict Energy Consumption"):
        if model is not None:
            # Prepare input data
            input_data = prepare_input_data()
            
            try:
                # Make prediction
                prediction = model.predict(input_data)
                
                # Display prediction
                st.success(f"### Predicted Energy Consumption: {prediction[0]:.2f} units")
                
                # Create a gauge chart to display the prediction
                fig, ax = plt.subplots(figsize=(10, 2))
                
                # Define gauge range (adjust based on your model's typical output range)
                min_energy = 0
                max_energy = 300
                
                # Normalize the prediction to gauge range
                norm_value = min(max(prediction[0], min_energy), max_energy)
                percentage = (norm_value - min_energy) / (max_energy - min_energy)
                
                # Plot the gauge
                ax.barh(0, percentage, height=0.6, color='green' if percentage < 0.5 else 'orange' if percentage < 0.8 else 'red')
                ax.barh(0, 1, height=0.6, color='lightgrey', alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([])
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels([f"{min_energy}", f"{min_energy + (max_energy-min_energy)*0.25:.0f}",
                                  f"{min_energy + (max_energy-min_energy)*0.5:.0f}", 
                                  f"{min_energy + (max_energy-min_energy)*0.75:.0f}", f"{max_energy}"])
                ax.set_xlabel("Energy Consumption (units)")
                ax.axvline(percentage, color='black', linestyle='--')
                
                st.pyplot(fig)
                
                # Display efficiency analysis
                st.subheader("Efficiency Analysis")
                if prediction[0] < 100:
                    st.info("✅ Energy consumption is low - system is operating efficiently.")
                elif prediction[0] < 200:
                    st.warning("⚠️ Moderate energy consumption - consider optimization strategies.")
                else:
                    st.error("🚨 High energy consumption - immediate efficiency measures recommended.")
                
                # Provide some insights
                st.subheader("Energy Saving Recommendations")
                recommendations = []
                
                if temperature < 19 or temperature > 25:
                    recommendations.append("Adjust temperature to between 20-24°C for optimal efficiency.")
                if hvac_usage == "On" and occupancy < 5:
                    recommendations.append("Consider turning off HVAC when building has low occupancy.")
                if lighting_usage == "On" and occupancy == 0:
                    recommendations.append("Turn off lights when the space is unoccupied.")
                if renewable_energy < 30:
                    recommendations.append("Increase renewable energy contribution to reduce carbon footprint and costs.")
                
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.write("No specific recommendations - current settings appear optimal.")
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("This may occur if the input features don't match what the model expects. Check that all required features are provided.")
        else:
            st.error("Model not loaded. Please check the model path.")

# Tab 2: Data Visualization
with tab2:
    st.header("Data Visualization")
    
    # Simulated historical data for visualization
    st.subheader("Simulated Energy Consumption Patterns")
    
    # Create some simulated data
    @st.cache_data
    def generate_sample_data():
        dates = pd.date_range(start=date_selected - datetime.timedelta(days=30), 
                             end=date_selected, freq='D')
        
        # Base consumption varies by temperature and weekday/weekend
        temps = np.random.normal(22, 5, size=len(dates))
        weekdays = [d.weekday() < 5 for d in dates]  # True for weekdays
        
        # Base consumption
        consumptions = 100 + 2 * temps + np.random.normal(0, 10, size=len(dates))
        # Add weekend effect (typically lower)
        consumptions = [c * 0.8 if not wd else c for c, wd in zip(consumptions, weekdays)]
        
        df = pd.DataFrame({
            'Date': dates,
            'Energy_Consumption': consumptions,
            'Temperature': temps,
            'Weekday': [d.strftime('%A') for d in dates]
        })
        return df
    
    sample_data = generate_sample_data()
    
    # Plot historical consumption
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_data['Date'], sample_data['Energy_Consumption'], marker='o', linestyle='-')
    ax.set_title('Energy Consumption - Last 30 Days')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy Consumption (units)')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to avoid overcrowding
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Correlation with temperature
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sample_data['Temperature'], sample_data['Energy_Consumption'])
        ax.set_title('Energy Consumption vs Temperature')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Energy Consumption (units)')
        ax.grid(True, alpha=0.3)
        
        # Add regression line
        z = np.polyfit(sample_data['Temperature'], sample_data['Energy_Consumption'], 1)
        p = np.poly1d(z)
        ax.plot(sample_data['Temperature'], p(sample_data['Temperature']), "r--", alpha=0.7)
        
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Weekday', y='Energy_Consumption', data=sample_data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        ax.set_title('Energy Consumption by Day of Week')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Energy Consumption (units)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Display input parameters vs typical range
    st.subheader("Current Input Parameters vs. Typical Range")
    
    parameter_ranges = {
        'Temperature (°C)': {'current': temperature, 'min': 15, 'max': 30, 'optimal_min': 20, 'optimal_max': 24},
        'Humidity (%)': {'current': humidity, 'min': 30, 'max': 70, 'optimal_min': 40, 'optimal_max': 60},
        'Occupancy (people)': {'current': occupancy, 'min': 0, 'max': 50, 'optimal_min': 5, 'optimal_max': 40},
        'Renewable Energy (%)': {'current': renewable_energy, 'min': 0, 'max': 100, 'optimal_min': 30, 'optimal_max': 100}
    }
    
    # Create a horizontal bar chart for each parameter
    fig, axes = plt.subplots(len(parameter_ranges), 1, figsize=(10, 8))
    
    for i, (param, values) in enumerate(parameter_ranges.items()):
        ax = axes[i]
        
        # Plot the full range
        ax.barh(param, values['max'] - values['min'], left=values['min'], height=0.3, color='lightgrey')
        
        # Plot the optimal range
        ax.barh(param, values['optimal_max'] - values['optimal_min'], left=values['optimal_min'], height=0.3, color='lightgreen')
        
        # Plot the current value
        ax.axvline(values['current'], color='red', linestyle='--')
        ax.text(values['current'], param, f" {values['current']}", va='center')
        
        # Set axis properties
        ax.set_xlim(min(values['min'], values['current']) - 5, max(values['max'], values['current']) + 5)
        
    plt.tight_layout()
    st.pyplot(fig)

# Tab 3: Model Information
with tab3:
    st.header("Model Information")
    
    st.subheader("About This Model")
    st.write("""
    This energy prediction model uses machine learning to forecast energy consumption based on environmental conditions, 
    building characteristics, and operational settings. The model was trained on historical data and can help facility 
    managers make informed decisions about energy usage.
    """)
    
    st.subheader("Feature Importance")
    
    # Create a simulated feature importance plot (since we don't have access to the actual model internals)
    feature_importance = {
        'Temperature': 0.25,
        'Occupancy': 0.20,
        'HVACUsage': 0.15,
        'SquareFootage': 0.12,
        'Time of Day': 0.10,
        'Humidity': 0.08,
        'LightingUsage': 0.05,
        'RenewableEnergy': 0.03,
        'Day of Week': 0.02
    }
    
    # Sort by importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(feature_importance.keys()), list(feature_importance.values()))
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.subheader("Using This App")
    st.write("""
    To use this application effectively:
    
    1. **Input Parameters**: Adjust the sliders and selectors in the sidebar to match current or projected conditions.
    2. **Predict**: Click the 'Predict' button to get an energy consumption estimate.
    3. **Analyze**: Use the visualizations to understand patterns and relationships in energy usage.
    4. **Optimize**: Follow the recommendations to optimize energy efficiency.
    
    For best results, ensure all inputs are within typical operating ranges for your facility.
    """)
    
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    
    # Display simulated model metrics
    col1.metric("R² Score", "0.89", "")
    col2.metric("RMSE", "15.7", "")
    col3.metric("MAE", "11.2", "")
    
    st.info("Note: These are sample performance metrics. Actual model performance may vary.")

# Footer
st.markdown("---")
st.markdown("Energy Consumption Prediction App | Created using Streamlit")