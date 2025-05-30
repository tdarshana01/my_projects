import numpy as np
import pandas as pd
import streamlit as st
import pickle
import json
import os

model = None
data_columns = None


def load_artifacts():
    global model, data_columns

    base_path = os.path.dirname(__file__)
    json_path = os.path.join(base_path, "artifacts", "Diamonds_columns.json")
    model_path = os.path.join(base_path, "artifacts", "Diamond_linear_model.pickle")

    with open(json_path, 'r') as f:
        data = json.load(f)
        data_columns = data['data_columns']

    with open(model_path, 'rb') as f:
        model = pickle.load(f)


def estimate_price(carat,cut,color,clarity,depth,table,x,y,z):
    load_artifacts()

    X = np.zeros(len(data_columns))
    X[0] = carat
    X[1] = cut
    X[2] = color
    X[3] = clarity
    X[4] = depth
    X[5] = table
    X[6] = x
    X[7] = y
    X[8] = z

    log_price =  model.predict([X])[0]
    return round(np.exp(log_price),2)

def main():
    st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")

    # Header HTML styling
    html_temp = """
        <style>
            .header-container {
                background: linear-gradient(to right, #00b4db, #0083b0);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .header-container h2 {
                color: white;
                text-align: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
        </style>
        <div class="header-container">
            <h2>💎 Diamond Price Prediction App 💎</h2>
            <div style='text-align: center; font-size: 18px; color: white; margin-top: 10px;'>
                Enter the diamond characteristics below and click <b>Predict</b> to estimate its price.
            </div>
        </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    cut_mapping = {
        'Fair':1,
        'Good':2,
        'Very Good':3,
        'Ideal':4,
        'Premium':5,
    }

    color_mapping = {'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7}
    clarity_mapping = {
        "I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8
    }

    carat = st.number_input("Carat",min_value=0.01, max_value=10.0, step=0.01, format="%.2f")
    depth = st.number_input("Depth",min_value=0.1, step = 0.1, format="%.1f")
    table = st.number_input("Table",min_value=0.1, step = 0.1, format="%.1f")
    x = st.number_input("x",min_value=0.01, step = 0.01, format="%.2f")
    y = st.number_input("y",min_value=0.01, step = 0.01, format="%.2f")
    z = st.number_input("z",min_value=0.01, step = 0.01, format="%.2f")
    cut = st.selectbox("Cut", list(cut_mapping.keys()))
    color = st.selectbox("Color", list(color_mapping.keys()))
    clarity = st.selectbox("Clarity", list(clarity_mapping.keys()))

    cut_value = cut_mapping[cut]
    color_value = color_mapping[color]
    clarity_value = clarity_mapping[clarity]

    result=""
    #if st.button("Predict"):
        #result=estimate_price(carat,cut_value,color_value,clarity_value,depth,table,x,y,z)
    if "show_result" not in st.session_state:
        st.session_state.show_result = False

    # Predict Button
    if st.button("Predict"):
        result=estimate_price(carat,cut_value,color_value,clarity_value,depth,table,x,y,z)

        st.session_state.show_result = True
        st.session_state.result = result

    # Display the result only if button was clicked
    if st.session_state.show_result:
        st.success(f"The estimated price of the diamond is **${st.session_state.result}**")

if __name__=='__main__':
    main()