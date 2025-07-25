import streamlit as st
import joblib
import pandas as pd
import numpy as np

def convert_time_to_hour(time_str):
    """Convert '1:00 PM' format to 24-hour integer (13)"""
    time, period = time_str.split()
    hour = int(time.split(':')[0])
    if period == "PM" and hour != 12:
        hour += 12
    elif period == "AM" and hour == 12:
        hour = 0
    return hour

def create_features(amount, hour):
    """Create the 4 features that the model expects"""
    return {
        'amount': amount,
        'hour': hour,
        'is_night': int(hour >= 22 or hour <= 6),
        'is_high_amount': int(amount > 50000)
    }

def get_risk_assessment(fraud_probability_percent):
    """Determine risk level and action based on fraud probability"""
    if fraud_probability_percent <= 50:
        return {
            'level': 'LOW',
            'action': 'ALLOWED',
            'color': 'success',
            'message': 'Transaction Approved',
            'recommendation': 'Process normally'
        }
    elif fraud_probability_percent <= 75:
        return {
            'level': 'MEDIUM', 
            'action': 'CONTACT_BANK',
            'color': 'warning',
            'message': 'Manual Review Required',
            'recommendation': 'Contact bank for verification'
        }
    else:
        return {
            'level': 'HIGH',
            'action': 'BLOCKED',
            'color': 'error', 
            'message': 'Transaction Blocked',
            'recommendation': 'Block and investigate'
        }

@st.cache_resource
def load_model():
    """Load the ML model and scaler"""
    try:
        model = joblib.load('model/fraud_model_random_forest.pkl')
        scaler = joblib.load('model/fraud_scaler.pkl')
        return model, scaler, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

def predict_fraud(features_dict, model, scaler):
    """Make fraud prediction using loaded model"""
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Use the correct feature order
        feature_order = ['amount', 'hour', 'is_night', 'is_high_amount']
        features = df[feature_order].values
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get fraud probability (class 1)
        if len(probability) == 2:
            fraud_probability = probability[1]
        else:
            fraud_probability = probability[0]
        
        return prediction, fraud_probability, True
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, False

# Streamlit App Configuration
st.set_page_config(layout="wide", page_title="Fraud Detection System")
st.title("FraudNet")
st.markdown("---")

# Load model at startup
model, scaler, model_loaded = load_model()

# Input Form
col1, col2 = st.columns(2)
with col1:
    sender_id = st.text_input("Sender's ID Number", placeholder="56921548", max_chars=8)
    amount = st.number_input("Amount (₱)", min_value=0.01, step=100.0, value=10000.00)

with col2:
    recipient_id = st.text_input("Recipient's ID Number", placeholder="69325719", max_chars=8)
    trans_type = st.selectbox("Transaction Type", 
                            ["Credit Purchase", "Debit Payment", "Money Transfer", "ATM Withdrawal"])

# Time Selection
time_of_day = st.selectbox("Time of Transaction", 
                        ["12:00 AM", "1:00 AM", "2:00 AM", "3:00 AM", "4:00 AM", 
                        "5:00 AM", "6:00 AM", "7:00 AM", "8:00 AM", "9:00 AM",
                        "10:00 AM", "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM",
                        "3:00 PM", "4:00 PM", "5:00 PM", "6:00 PM", "7:00 PM",
                        "8:00 PM", "9:00 PM", "10:00 PM", "11:00 PM"])

# Analyze Button
if st.button("Analyze Transaction", type="primary"):
    if not model_loaded:
        st.error("Model not loaded! Cannot perform prediction.")
        st.stop()
    
    # Basic validation
    if not sender_id or sender_id.strip() == "":
        st.error("Sender's ID Number is required")
        st.stop()
    
    # Check sender ID length
    if len(sender_id.strip()) != 8:
        st.error("Sender's ID Number must be exactly 8 digits")
        st.stop()
    
    # Check if recipient ID is required for Money Transfer
    if trans_type == "Money Transfer":
        if not recipient_id or recipient_id.strip() == "":
            st.error("Recipient's ID Number is required for Money Transfer")
            st.stop()
        if len(recipient_id.strip()) != 8:
            st.error("Recipient's ID Number must be exactly 8 digits")
            st.stop()
    
    # Convert time to hour and create features
    hour = convert_time_to_hour(time_of_day)
    features = create_features(amount, hour)

    # Make prediction
    with st.spinner('Analyzing transaction...'):
        prediction, fraud_probability, prediction_success = predict_fraud(features, model, scaler)
    
    if prediction_success and prediction is not None:
        # Convert probability to percentage
        fraud_probability_percent = fraud_probability * 100
        
        # Get risk assessment
        risk_assessment = get_risk_assessment(fraud_probability_percent)
        
        # Display results
        st.markdown("## Analysis Results")
        
        # Create columns for result and risk meter
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            if risk_assessment['action'] == 'ALLOWED':
                st.success(f"""
                ✅ **{risk_assessment['message']}**
                
                **Fraud Probability:** {fraud_probability_percent:.2f}%  
                **Risk Level:** {risk_assessment['level']}  
                **Amount:** ₱{amount:,.2f}  
                **Transaction Time:** {time_of_day}  
                **Action:** {risk_assessment['recommendation']}
                """)
                
            elif risk_assessment['action'] == 'CONTACT_BANK':
                st.warning(f"""
                ⚠️ **{risk_assessment['message']}**
                
                **Fraud Probability:** {fraud_probability_percent:.2f}%  
                **Risk Level:** {risk_assessment['level']}  
                **Amount:** ₱{amount:,.2f}  
                **Transaction Time:** {time_of_day}  
                **Action:** {risk_assessment['recommendation']}
                """)
                
            else:  # BLOCKED
                st.error(f"""
                🚫 **{risk_assessment['message']}**
                
                **Fraud Probability:** {fraud_probability_percent:.2f}%  
                **Risk Level:** {risk_assessment['level']}  
                **Amount:** ₱{amount:,.2f}  
                **Transaction Time:** {time_of_day}  
                **Action:** {risk_assessment['recommendation']}
                """)
        
        with result_col2:
            # Risk meter visualization
            st.markdown("### Risk Meter")
            
            # Color-coded progress bar based on risk level
            if fraud_probability_percent <= 50:
                # Green for low risk
                st.success("🟢 LOW RISK")
            elif fraud_probability_percent <= 75:
                # Yellow for medium risk
                st.warning("🟡 MEDIUM RISK")
            else:
                # Red for high risk
                st.error("🔴 HIGH RISK")
            
            # Progress bar
            st.progress(fraud_probability_percent / 100)
            st.markdown(f"**{fraud_probability_percent:.1f}%** Fraud Probability")
            
            # Risk threshold indicators
            st.markdown("""
            **Thresholds:**
            - 0-50%: Allow
            - 50-75%: Review  
            - 76-100%: Block
            """)
        
        # Transaction Details
        st.markdown("---")
        st.markdown("### Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Transaction Information:**
            - Sender ID: {sender_id}
            - Recipient ID: {recipient_id if recipient_id else 'N/A'}
            - Transaction Type: {trans_type}
            - Amount: ₱{amount:,.2f}
            - Time: {time_of_day} (Hour: {hour})
            """)
        
        with col2:
            is_night = hour >= 22 or hour <= 6
            is_high_amount = amount > 50000
            
            st.info(f"""
            **Model Analysis:**
            - Amount: ₱{amount:,.2f}
            - Hour: {hour}
            - Night Transaction: {'Yes' if is_night else 'No'}
            - High Amount: {'Yes' if is_high_amount else 'No'}
            - Fraud Probability: {fraud_probability_percent:.2f}%
            - Final Decision: {risk_assessment['level']} Risk
            """)

# System Status
st.markdown("---")
st.markdown("## System Configuration")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Low Risk", "0-50%", "Allowed")
with col2:
    st.metric("Medium Risk", "50-75%", "Contact Bank")
with col3:
    st.metric("High Risk", "76-100%", "Blocked")
with col4:
    if model_loaded:
        st.success("**Model Status**\n✅ Loaded")
    else:
        st.error("**Model Status**\n❌ Failed")

# Model Information
with st.expander("How the System Works"):
    st.markdown("""
    **Risk Assessment Process:**
    
    **Risk Thresholds:**
    - **0-50% Risk**: Transaction automatically approved
    - **50-75% Risk**: Manual review required - contact bank for verification
    - **76-100% Risk**: Transaction blocked by system
    
    **Model Features:**
    
    1. **amount** - Transaction amount in Philippine Peso
    2. **hour** - Hour of transaction (0-23, 24-hour format)  
    3. **is_night** - Binary flag for night time (22:00-06:00)
    4. **is_high_amount** - Binary flag for amounts over ₱50,000
    
    **How It Works:**
    - The system analyzes transaction patterns using machine learning
    - It outputs a probability (0-100%) that a transaction is fraudulent
    - Only this probability determines the action taken
    - No hard-coded rules override the model's decision
    """)

# Show feature importance if model loaded
if model_loaded and model and hasattr(model, 'feature_importances_'):
    with st.expander("Model Feature Importance"):
        feature_names = ['amount', 'hour', 'is_night', 'is_high_amount']
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        st.markdown("**Feature contributions to fraud detection:**")
        for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{feature}**: {importance:.3f} ({importance*100:.1f}%)")
            st.progress(importance)

st.markdown("---")