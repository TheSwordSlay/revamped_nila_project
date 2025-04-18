import streamlit as st
import pandas as pd
from SVM import SVM
from SVMMi import SVMmi

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# App title and description
st.title("Sentiment Analysis with SVM")
st.markdown("""
Upload your Excel/CSV file and choose processing method.
The file should contain 'Content' and 'Skor' columns.
""")

# File upload section
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx"],
    accept_multiple_files=False
)

# Method selection
method = st.radio(
    "Select processing method:",
    ("SVM", "SVM with Mutual Information"),
    horizontal=True
)

if uploaded_file is not None:
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Validate required columns
        if not all(col in df.columns for col in ['Content', 'Skor']):
            st.error("File must contain 'Content' and 'Skor' columns")
        else:
            # Process based on selected method
            with st.spinner('Processing...'):
                if method == "SVM":
                    results_df, accuracy = SVM(df)
                else:
                    results_df, accuracy = SVMmi(df)

            # Display results
            st.success("Processing complete!")
            st.subheader("Results")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(results_df, use_container_width=True)
            with col2:
                st.metric("Accuracy", f"{accuracy:.2%}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")