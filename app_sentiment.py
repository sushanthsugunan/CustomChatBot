import streamlit as st
import pandas as pd
from chatbot_neo import process_csv

st.title("Sentiment and Tone Analysis with GPT-Neo")
st.write("Upload a CSV file with sentences to analyze sentiment and tone.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df_result = process_csv(uploaded_file)
    if isinstance(df_result, str):
        st.error(df_result)
    else:
        st.write("Analysis Results:")
        st.dataframe(df_result)
