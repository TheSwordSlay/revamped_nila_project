import streamlit as st
import pandas as pd
from SVM import SVM
from SVMMi import SVMmi
import plotly.express as px
import nltk
try:
    nltk.download('stopwords')  # For English stopwords
    nltk.download('punkt')  # Required for some NLTK operations
    # Verify Indonesian stopwords are available
    if 'indonesian' not in nltk.corpus.stopwords.fileids():
        st.error("Indonesian stopwords not found in NLTK. Using English stopwords instead.")
except Exception as e:
    st.error(f"Failed to download NLTK data: {str(e)}")

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

@st.cache_data
def returnSVM(df):
    return SVM(df)

@st.cache_data
def returnSVMMI(df):
    return SVMmi(df)

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
                    results_df, metrics = returnSVM(df)
                else:
                    results_df, metrics = returnSVMMI(df)

            # Display results
            st.success("Processing complete!")
            st.subheader("Results")
            
            col1, col2 = st.columns([3, 2])
            with col1:
                st.dataframe(results_df, use_container_width=True)
            with col2:
                st.subheader("Performance Metrics")
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                st.metric("Precision", f"{metrics['precision']:.2%}")
                st.metric("Recall", f"{metrics['recall']:.2%}")
                st.metric("F1-Score", f"{metrics['f1']:.2%}")
            
            st.subheader("Confusion Matrix")
            fig = px.imshow(
                metrics['confusion_matrix'],
                labels=dict(x="Predicted", y="Actual"),
                text_auto=True,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")