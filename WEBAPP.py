import numpy as np
import pandas as pd
import streamlit as st
import webbrowser
import joblib  # Change to joblib for loading the model

# Load the model
try:
    with open('iris_model.joblib', 'rb') as file:
        RFC = joblib.load(file)
    print("Model loaded successfully:", type(RFC))
except Exception as e:
    st.error(f"Error loading model: {e}")
    RFC = None  # Ensure RFC is None if loading fails

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def predict():
    if RFC is None:
        st.error("Model not loaded.")
        return
    
    try:
        row = np.array([sepal_length, sepal_width, petal_length, petal_width])
        x = pd.DataFrame([row], columns=columns)
        prediction = RFC.predict(x)[0]

        if prediction == 0:
            st.success("It is SETOSA")
        elif prediction == 1:
            st.success("It is VERSICOLOR")
        elif prediction == 2:
            st.success("It is VIRGINICA")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

#### SIDEBAR ####
st.sidebar.title("USER INPUTS")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.30, 7.90)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.20, 4.40)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.00, 6.90)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.10, 2.50)
st.sidebar.button('SUBMIT', on_click=predict)

#### MAIN PAGE ####
input_data = pd.DataFrame({
    'sepal_length_(cm)': [sepal_length],
    'sepal_width_(cm)': [sepal_width],
    'petal_length_(cm)': [petal_length],
    'petal_width_(cm)': [petal_width]
})

expected_data = {
    'Setosa': 0,
    'Versicolor': 1,
    'Virginica': 2
}
st.markdown("# Welcome Home! Discover More, Dream Bigger! :book:")
st.subheader("Input Features")
st.dataframe(input_data)

st.subheader("Expected Outcomes")
st.dataframe(pd.DataFrame(list(expected_data.items()), columns=['Species', 'Label']))

if st.button("More on this"):
    webbrowser.open_new_tab("https://www.linkedin.com/posts/garvit-gupta-87875a226_data-pre-processing-activity-7162410464151322624-AhdR?utm_source=share&utm_medium=member_desktop")
