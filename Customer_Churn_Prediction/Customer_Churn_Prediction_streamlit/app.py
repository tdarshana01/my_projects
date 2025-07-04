import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import json

model = None
data_columns = None
scale = None

def load_artifacts():
    global model, data_columns, scale

    base_path = os.path.dirname(__file__)
    json_path = os.path.join(base_path,'artifacts','Churn_cols.json')
    model_path = os.path.join(base_path,'artifacts','Churn_Prediction_model.pickle')
    scale_path = os.path.join(base_path,'artifacts','scaler_1.pickle')

    with open(json_path,'r') as f:
        data = json.load(f)
        data_columns = data['data_columns']

    with open(model_path,'rb') as f:
        model = pickle.load(f)

    with open(scale_path,'rb') as f:
        scale = joblib.load(f)

def estimate_churn(gender,seniorcitizen,partner,dependents,tenure,phoneservice,multiplelines,internetservice,onlinesecurity,onlinebackup,deviceprotection,techsupport,streamingtv,streamingmovies,contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges):
    load_artifacts()

    X = np.zeros(len(data_columns))
    fields = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'SeniorCitizen': seniorcitizen,
        'PhoneService': phoneservice,
        'MultipleLines': multiplelines,
        'InternetService': internetservice,
        'OnlineSecurity': onlinesecurity,
        'OnlineBackup': onlinebackup,
        'DeviceProtection': deviceprotection,
        'TechSupport': techsupport,
        'StreamingTV': streamingtv,
        'StreamingMovies': streamingmovies,
        'Contract': contract,
        'PaperlessBilling': paperlessbilling,
        'PaymentMethod': paymentmethod
    }

    for feature,value in fields.items():
        column_name = f"{feature}_{value}"
        if column_name in data_columns:
            index = data_columns.index(column_name)
            X[index] = 1

    numerical_input = np.array([[tenure, monthlycharges, totalcharges]])
    scaled_values = scale.transform(numerical_input)[0]

    X[data_columns.index('tenure')] = scaled_values[0]
    X[data_columns.index('MonthlyCharges')] = scaled_values[1]
    X[data_columns.index('TotalCharges')] = scaled_values[2]

    prediction = model.predict([X])[0]
    result = ''
    if prediction == 1:
        result = 'more'
    else:
        result = 'less'

    probability = model.predict_proba([X])[0][1]
    return result, probability

def main():
    html_temp = """
        <div style="background: linear-gradient(to right, #11998e, #38ef7d); padding: 15px 10px; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0,0,0,0.2);">
            <h2 style="color: white; text-align: center; font-family: 'Segoe UI', sans-serif; margin: 0;">🚀 Customer Churn Prediction App</h2>
            <p style="color: #f0fdf4; text-align: center; font-size: 14px; margin-top: 5px;">Predict which customers are most likely to churn and take action early.</p>
        </div>
        <br>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("""
    Customer churn refers to the phenomenon where a customer stops doing business or ends their relationship with a company.  
    Understanding and predicting customer churn is crucial for businesses, especially in subscription-based industries like telecommunications, banking, and SaaS platforms.  

    This app allows users to upload a customer dataset and instantly receive churn predictions.  
    By analyzing features such as contract type, payment method, tenure, and internet service,  
    a machine learning model predicts whether a customer is likely to churn.  
    """)

    st.info("Upload a dataset and get your predictions!")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        input_df_1 = input_df.drop('customerID',axis=1)

        expected_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
           'MonthlyCharges', 'TotalCharges']

        input_df_1 = input_df_1[expected_columns]

        base_path = os.path.dirname(__file__)
        pipeline_path = os.path.join(base_path,'artifacts','pipeline.pickle')
        pipeline = joblib.load(pipeline_path)
        pred_proba = pipeline.predict_proba(input_df_1)[:,1]
        prediction = (pred_proba > 0.56).astype(int)
        prediction = pd.Series(prediction)
        prediction = prediction.apply(lambda x:'Yes' if x==1 else 'No')
        pred_proba = pd.Series(pred_proba)

        result = pd.DataFrame({
            'customerID': input_df['customerID'],
            'Churn': prediction,
            'Churn Probability': (pred_proba*100).apply(lambda x:f'{x:.2f}%')
        })

        st.dataframe(result)

        no_churn = len(result[result['Churn'] == 'Yes'])
        total = len(result)
        percentage = np.round((no_churn / total * 100),2)

        st.info(f"{percentage}% of customers likely to churn")
        st.warning("Here we used the threshold for churn as 56% to flag customers as likely to churn, ensuring a better trade-off between false positives and missed churners.")

    else:
        st.warning("Please upload a dataset and get your predictions!")

    with st.expander("Get individual predictions"):
        st.info("Enter the attributes of customer below and click to estimate the Churn of the customer")

        gender = st.selectbox("Gender",['Male','Female'])
        seniorcitizen = st.selectbox("SeniorCitizen",['Yes','No'])
        partner = st.selectbox("Partner",['Yes','No'])
        dependents = st.selectbox("Dependents",['Yes','No'])
        tenure = st.number_input("Tenure",min_value=1,step=1)
        phoneservice = st.selectbox("PhoneService",['Yes','No'])
        multiplelines = st.selectbox("MultipleLines",['Yes','No','No Phone Service'])
        internetservice = st.selectbox("InternetService",['DSL','Fiber optic','No'])
        onlinesecurity = st.selectbox("OnlineSecurity",['Yes','No','No Internet Service'])
        onlinebackup = st.selectbox("OnlineBackup",['Yes','No','No Internet Service'])
        deviceprotection = st.selectbox("DeviceProtection",['Yes','No','No Internet Service'])
        techsupport = st.selectbox("TechSupport",['Yes','No','No Internet Service'])
        streamingtv = st.selectbox("StreamingTV",['Yes','No','No Internet Service'])
        streamingmovies = st.selectbox("StreamingMovies",['Yes','No','No Internet Service'])
        contract = st.selectbox("Contract",['Month-to-month','One year','Two year'])
        paperlessbilling = st.selectbox("PaperlessBilling",['Yes','No'])
        paymentmethod = st.selectbox("PaymentMethod",['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
        monthlycharges = st.number_input("MonthlyCharges",min_value=1.00,format="%.2f",step=0.01)
        totalcharges = st.number_input("TotalCharges",min_value=1.00,format="%.2f",step=0.01)

        if "show_result" not in st.session_state:
            st.session_state.show_result = False

        col1,col2 = st.columns(2)

        # Predict Button
        with col1:
            if st.button("Predict"):
                result,probability = estimate_churn(gender,seniorcitizen,partner,dependents,tenure,phoneservice,multiplelines,internetservice,onlinesecurity,onlinebackup,deviceprotection,techsupport,streamingtv,streamingmovies,contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges)
                st.session_state.show_result = True
                st.session_state.result = result
                st.session_state.probability = probability

        # Display the result only if button was clicked
        if st.session_state.show_result:
            st.success(f"**The customer is {st.session_state.result} likely to churn**")
            st.success(f"**The probability of churn is {st.session_state.probability:.2%}**")

        # Clear All Button
        with col2:
            if st.button("Clear All"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

    with st.expander("Data"):
        st.write('Sample data')
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'data', 'Customer_churn_data.csv')
        df = pd.read_csv(file_path)
        df


if __name__ == '__main__':
    main()
