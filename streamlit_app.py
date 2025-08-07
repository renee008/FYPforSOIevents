import streamlit as st
import pandas as pd
import joblib # For loading the scalers and models
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier
import lightgbm as lgb # Import LightGBM
import xgboost as xgb # Import XGBoost
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Import VADER
import nltk # Import nltk
import numpy as np # Import numpy for array handling
import matplotlib.pyplot as plt # For plotting contributions

# For URL fetching and parsing
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import fitz  # PyMuPDF
import io
from boilerpy3.extractors import ArticleExtractor # Import boilerpy3

# --- Custom CSS for a modern, cohesive, and aesthetically pleasing design ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* --- Main App Container --- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px; /* Constrain width for a more focused layout */
    }
    
    /* --- Header Styling --- */
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50; /* Dark blue/gray */
    }
    h2 {
        font-size: 1.75rem;
        font-weight: 600;
        color: #34495e; /* Slightly lighter dark blue/gray */
    }
    
    /* --- Streamlit's Main Container with a card-like effect --- */
    .st-emotion-cache-1833z0x { /* The main app container class */
        background-color: #f0f2f6; /* Light gray background */
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        padding: 3rem;
        margin-bottom: 2rem;
    }
    
    /* --- Button Styling --- */
    .stButton>button {
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.2s ease-in-out;
        border: none;
        background-color: #3498db; /* Blue */
        color: white;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* --- Text Input & Text Area Styling --- */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        padding: 8px 12px;
        border: 1px solid #bdc3c7;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        transition: border-color 0.2s ease-in-out;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
    }
    
    /* --- Tabs Container Styling --- */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
        color: #7f8c8d; /* Gray */
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        color: #3498db; /* Blue */
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #ecf0f1;
        border-radius: 12px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: transparent;
        border-radius: 8px;
        transition: background-color 0.2s;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* --- Expander Styling --- */
    .streamlit-expanderHeader {
        background-color: #ecf0f1;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        color: #2c3e50;
        transition: background-color 0.2s;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .streamlit-expanderHeader:hover {
        background-color: #dce1e7;
    }

    /* --- Metric Styling --- */
    [data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 700;
        color: #27ae60; /* Green for success */
    }
    [data-testid="stMetricLabel"] {
        color: #7f8c8d;
    }
    
    /* --- Info, Warning, Error boxes --- */
    .st-emotion-cache-1629p8f { /* Info */
        border-left: 5px solid #3498db;
        border-radius: 8px;
    }
    .st-emotion-cache-14u4v27 { /* Warning */
        border-left: 5px solid #f1c40f;
        border-radius: 8px;
    }
    .st-emotion-cache-14n91p4 { /* Error */
        border-left: 5px solid #e74c3c;
        border-radius: 8px;
    }

    /* --- Hide number input arrows --- */
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type=number] {
        -moz-appearance: textfield; /* Firefox */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- NLTK Data Download (Crucial for VADER) ---
@st.cache_resource # Cache the download to run only once
def download_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError: # Catch LookupError for missing data
        nltk.download('vader_lexicon')

download_nltk_vader()

# --- Configuration ---
st.set_page_config(page_title="Credit Rating & Sentiment Predictor", page_icon="📈", layout="wide") # Changed to wide layout and new icon

# --- Define Feature Columns (Consistent with your MLtraining.py) ---
# This list has 20 financial features
FINANCIAL_COLS = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin',
    'returnOnAssets', 'returnOnEquity', 'assetTurnover',
    'fixedAssetTurnover', 'debtRatio', 'effectiveTaxRate',
    'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare',
    'enterpriseValueMultiple', 'payablesTurnover','operatingCashFlowPerShare', 'operatingCashFlowSalesRatio'
]

# Define categories for financial inputs for better UI
FINANCIAL_CATEGORIES = {
    "Liquidity Ratios": ['currentRatio', 'quickRatio', 'cashRatio'],
    "Profitability & Margins": ['netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin', 'operatingCashFlowSalesRatio'],
    "Asset Efficiency": ['daysOfSalesOutstanding', 'assetTurnover', 'fixedAssetTurnover', 'payablesTurnover'],
    "Leverage & Debt": ['debtRatio'],
    "Return Ratios": ['returnOnAssets', 'returnOnEquity'],
    "Cash Flow Metrics": ['freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare', 'operatingCashFlowPerShare'],
    "Other Key Metrics": ['enterpriseValueMultiple', 'effectiveTaxRate']
}

# Define min and max values for financial inputs for better user experience
FINANCIAL_INPUT_BOUNDS = {
    'currentRatio': (0.0, 50.0), 'quickRatio': (0.0, 50.0), 'cashRatio': (0.0, 50.0),
    'daysOfSalesOutstanding': (0.0, 500.0), 'netProfitMargin': (-2.0, 2.0),
    'pretaxProfitMargin': (-2.0, 2.0), 'grossProfitMargin': (-2.0, 2.0),
    'returnOnAssets': (-2.0, 2.0), 'returnOnEquity': (-5.0, 5.0),
    'assetTurnover': (0.0, 10.0), 'fixedAssetTurnover': (0.0, 20.0),
    'debtRatio': (0.0, 1.0), 'effectiveTaxRate': (-2.0, 1.0),
    'freeCashFlowOperatingCashFlowRatio': (-5.0, 5.0), 'freeCashFlowPerShare': (-100.0, 100.0),
    'cashPerShare': (-100.0, 100.0), 'enterpriseValueMultiple': (0.0, 50.0),
    'payablesTurnover': (0.0, 100.0), 'operatingCashFlowPerShare': (-100.0, 100.0),
    'operatingCashFlowSalesRatio': (-2.0, 2.0)
}


SENTIMENT_COLS = ['Avg_Positive', 'Avg_Neutral', 'Avg_Negative', 'Avg_Compound']
ALL_COLS = FINANCIAL_COLS + SENTIMENT_COLS

# --- Define a consistent order for credit ratings ---
RATING_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']

# --- Model and Scaler Loading ---
@st.cache_resource # Cache the models and scalers to avoid reloading on every rerun
def load_models_and_scalers():
    models = {}
    scalers = {}
    label_encoder = None # Initialize label_encoder here

    try:
        # Load CatBoost Model A (Financial Only) and its scaler (renee_scaler_financial.pkl)
        cat_model_A = CatBoostClassifier()
        cat_model_A.load_model('CatboostML.modelA.cbm')
        models['CatBoost Model A (Financial Only)'] = cat_model_A
        scalers['CatBoost Model A (Financial Only)'] = joblib.load('renee_scaler_financial.pkl')

        # Load CatBoost Model B (Financial + Sentiment) and its scaler (renee_scaler_all.pkl)
        cat_model_B = CatBoostClassifier()
        cat_model_B.load_model('CatboostML.modelB.cbm')
        models['CatBoost Model B (Financial + Sentiment)'] = cat_model_B
        scalers['CatBoost Model B (Financial + Sentiment)'] = joblib.load('renee_scaler_all.pkl')

        # Load RandomForest Model A (Financial Only) and its scaler (ath_scaler_financial.pkl)
        rf_model_A = joblib.load('ath_modelA_randomforest.pkl')
        models['RandomForest Model A (Financial Only)'] = rf_model_A
        scalers['RandomForest Model A (Financial Only)'] = joblib.load('ath_scaler_financial.pkl')

        # Load RandomForest Model B (Financial + Sentiment) and its scaler (ath_scaler_all.pkl)
        rf_model_B = joblib.load('ath_modelB_randomforest.pkl')
        models['RandomForest Model B (Financial + Sentiment)'] = rf_model_B
        scalers['RandomForest Model B (Financial + Sentiment)'] = joblib.load('ath_scaler_all.pkl')

        # Load LightGBM Model A (Financial Only) and its scaler (ralph_scaler_financial.pkl)
        lgbm_model_A = lgb.Booster(model_file='LightGBM.modelA.txt')
        models['LightGBM Model A (Financial Only)'] = lgbm_model_A
        scalers['LightGBM Model A (Financial Only)'] = joblib.load('ralph_scaler_financial.pkl')

        # Load LightGBM Model B (Financial + Sentiment) and its scaler (ralph_scaler_all.pkl)
        lgbm_model_B = lgb.Booster(model_file='LightGBM.modelB.txt')
        models['LightGBM Model B (Financial + Sentiment)'] = lgbm_model_B
        scalers['LightGBM Model B (Financial + Sentiment)'] = joblib.load('ralph_scaler_all.pkl')

        # Load XGBoost Model A (Financial Only) and its scaler (yu_pin_scaler_financial.pkl)
        xgb_model_A = joblib.load('xgb_model_A_financial.pkl')
        models['XGBoost Model A (Financial Only)'] = xgb_model_A
        scalers['XGBoost Model A (Financial Only)'] = joblib.load('yu_pin_scaler_financial.pkl')

        # Load XGBoost Model B (Financial + Sentiment) and its scaler (yu_pin_scaler_all.pkl)
        xgb_model_B = joblib.load('xgb_model_B_financial_sentiment.pkl')
        models['XGBoost Model B (Financial + Sentiment)'] = xgb_model_B
        scalers['XGBoost Model B (Financial + Sentiment)'] = joblib.load('yu_pin_scaler_all.pkl')

        # Load the LabelEncoder for credit ratings (used by XGBoost, RandomForest, LightGBM)
        label_encoder = joblib.load('rating_label_encoder.pkl')


        st.success("All individual models and scalers loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler file: {e}. Please ensure all model and scaler files are in the same directory.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()
    return models, scalers, label_encoder

# Load models and scalers at the start of the app
models, scalers, label_encoder = load_models_and_scalers()

# --- Prediction Function for a Single Model ---
def _predict_single_model(model, scaler, input_df, feature_columns, label_encoder) -> tuple:
    """
    Predicts the credit rating and its probabilities using a single given model and scaler.
    Returns (predicted_rating_str, probabilities_dict).
    """
    try:
        input_df_reindexed = input_df[feature_columns]
        
        # Debugging: Check for NaNs in input_df_reindexed
        if input_df_reindexed.isnull().any().any():
            st.error(f"NaN values detected in input features for model {type(model).__name__}. Please check your inputs.")
            return "Prediction failed (NaN in input).", {}

        scaled_data = scaler.transform(input_df_reindexed)

        # Handle prediction based on model type
        if isinstance(model, CatBoostClassifier):
            # CatBoost predict can return a 1-element array, ensure it's a scalar string
            predicted_rating_raw = model.predict(scaled_data)[0]
            if isinstance(predicted_rating_raw, np.ndarray): # If it's still a numpy array (e.g., array(['A']))
                predicted_rating_str = predicted_rating_raw.item() # Extract the scalar string
            else: # It's already a string
                predicted_rating_str = str(predicted_rating_raw)
            
            probabilities = model.predict_proba(scaled_data)[0]
            class_names = model.classes_ # CatBoost stores class names
        elif isinstance(model, RandomForestClassifier) or isinstance(model, xgb.XGBClassifier):
            raw_prediction = model.predict(scaled_data)[0]
            # If raw_prediction is a string (e.g., 'A'), convert it to int index using label_encoder
            if isinstance(raw_prediction, str):
                predicted_class_idx = label_encoder.transform([raw_prediction])[0]
            else: # Otherwise, it's already a numerical index
                predicted_class_idx = raw_prediction
            
            predicted_rating_str = label_encoder.inverse_transform([int(predicted_class_idx)])[0] # Convert to string
            probabilities = model.predict_proba(scaled_data)[0]
            class_names = label_encoder.classes_ # Use label encoder classes for consistency
        elif isinstance(model, lgb.Booster):
            probabilities = model.predict(scaled_data)[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_rating_str = label_encoder.inverse_transform([predicted_class_idx])[0]
            class_names = label_encoder.classes_ # Use label encoder classes for consistency
        else:
            return "Unsupported model type.", {}

        probabilities_dict = dict(zip(class_names, probabilities))

        return str(predicted_rating_str), probabilities_dict

    except Exception as e:
        st.error(f"Error during single model prediction ({type(model).__name__}): {e}")
        return "Prediction failed.", {}

# --- Sentiment Analysis Function (Updated to handle URL) ---
# Keywords for filtering relevant sentences (consistent with part6.py)
KEYWORDS = [
    'revenue', 'earnings', 'profit', 'loss', 'income', 'debt', 'dividend',
    'forecast', 'growth', 'decline', 'investment', 'shareholder', 'stock',
    'market', 'guidance', 'cash flow', 'valuation', 'credit', 'expenses',
    'capital', 'return', 'liability', 'asset', 'equity', 'margin', 'rating',
    'billion', 'million'
]

@st.cache_data(show_spinner=False) # Cache the results of web scraping and sentiment analysis
def fetch_and_analyze_single_url(url: str) -> tuple[dict, str]:
    """
    Fetches content from a single URL, extracts relevant text, and performs sentiment analysis.
    Returns (sentiment_scores_dict, extracted_text_or_error_message).
    """
    if not url:
        return ({}, "No URL provided.")

    text = ""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        if url.lower().endswith(".pdf"):
            pdf_file = io.BytesIO(response.content)
            with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
        else:
            # Use boilerpy3 for robust article extraction from HTML
            extractor = ArticleExtractor()
            document = extractor.get_doc(response.text)
            text = document.content
            
    except requests.exceptions.RequestException as e:
        return ({}, f"Error fetching URL '{url}': {e}")
    except Exception as e:
        return ({}, f"Error parsing content from URL '{url}': {e}")

    if not text:
        return ({}, f"No readable content found at URL '{url}' or content was too short.")

    # Apply keyword filtering and sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant_sentences = [s for s in sentences if any(kw in s.lower() for kw in KEYWORDS) and 5 <= len(s.split()) <= 60]

    if not relevant_sentences:
        return ({}, f"No relevant sentences found in the article from '{url}' for sentiment analysis.")

    pos_scores, neu_scores, neg_scores, compound_scores = [], [], [], []
    for sentence in relevant_sentences:
        sentiment = analyzer.polarity_scores(sentence)
        pos_scores.append(sentiment['pos'])
        neu_scores.append(sentiment['neu'])
        neg_scores.append(sentiment['neg'])
        compound_scores.append(sentiment['compound'])

    # Calculate average scores for this single URL
    avg_pos = sum(pos_scores) / len(pos_scores)
    avg_neu = sum(neu_scores) / len(neu_scores)
    avg_neg = sum(neg_scores) / len(neg_scores)
    avg_comp = sum(compound_scores) / len(compound_scores)

    # Determine category for this single URL
    if avg_comp >= 0.3: # Using user-defined threshold
        category = "Positive"
    elif avg_comp <= -0.3: # Using user-defined threshold
        category = "Negative"
    else:
        category = "Neutral"

    sentiment_result = {
        'Avg_Positive': avg_pos,
        'Avg_Neutral': avg_neu,
        'Avg_Negative': avg_neg,
        'Avg_Compound': avg_comp,
        'category': category # Include category in the returned dict
    }
    return sentiment_result, "\n\n".join(relevant_sentences)


def analyze_multiple_urls_sentiment(urls: list[str]) -> tuple[dict, dict]:
    """
    Analyzes sentiment from multiple URLs and returns average scores and detailed results.
    Returns (overall_sentiment_dict, detailed_results_dict).
    """
    all_pos_scores, all_neu_scores, all_neg_scores, all_comp_scores = [], [], [], []
    detailed_results = {}
    for url in urls:
        if url.strip(): # Ensure URL is not empty
            sentiment_for_url, _ = fetch_and_analyze_single_url(url.strip()) # Don't need extracted_text here
            if sentiment_for_url and 'Avg_Compound' in sentiment_for_url: # Successfully got sentiment scores
                all_pos_scores.append(sentiment_for_url['Avg_Positive'])
                all_neu_scores.append(sentiment_for_url['Avg_Neutral'])
                all_neg_scores.append(sentiment_for_url['Avg_Negative'])
                all_comp_scores.append(sentiment_for_url['Avg_Compound'])
                detailed_results[url] = {"status": "Success", "sentiment": sentiment_for_url}
            else:
                detailed_results[url] = {"status": "Failed", "error": _} # _ contains error message
    
    if not all_comp_scores: # No URLs were successfully analyzed
        return ({'Avg_Positive': 0.0, 'Avg_Neutral': 1.0, 'Avg_Negative': 0.0, 'Avg_Compound': 0.0, 'category': 'Neutral'}, detailed_results) # Default category for no successful URLs
    
    # Calculate overall averages
    overall_avg_pos = sum(all_pos_scores) / len(all_pos_scores)
    overall_avg_neu = sum(all_neu_scores) / len(all_neu_scores)
    overall_avg_neg = sum(all_neg_scores) / len(all_neg_scores)
    overall_avg_comp = sum(all_comp_scores) / len(all_comp_scores)

    # Apply the user-defined thresholds for overall category
    if overall_avg_comp >= 0.3: # Changed from 0.05 to 0.3
        overall_category = "Positive"
    elif overall_avg_comp <= -0.3: # Changed from -0.05 to -0.3
        overall_category = "Negative"
    else:
        overall_category = "Neutral"

    overall_sentiment_result = {
        'Avg_Positive': overall_avg_pos,
        'Avg_Neutral': overall_avg_neu,
        'Avg_Negative': overall_avg_neg,
        'Avg_Compound': overall_avg_comp,
        'category': overall_category
    }
    return overall_sentiment_result, detailed_results

# --- Credit Rating Definitions ---
CREDIT_RATING_DEFINITIONS = {
    'AAA': "Highest quality, lowest risk. Extremely strong capacity to meet financial commitments.",
    'AA': "Very high quality, very low risk. Very strong capacity to meet financial commitments.",
    'A': "High quality, low risk. Strong capacity to meet financial commitments, but somewhat susceptible to adverse economic conditions.",
    'BBB': "Adequate quality, moderate risk. Adequate capacity to meet financial commitments, but more vulnerable to adverse economic conditions.",
    'BB': "Speculative, high risk. Faces major ongoing uncertainties or exposure to adverse business, financial, or economic conditions.",
    'B': "Highly speculative, very high risk. Financial commitments are currently being met, but capacity is vulnerable to a material change in circumstances.",
    'CCC': "Substantial risk. Currently vulnerable, and dependent on favorable business, financial, and economic conditions to meet its financial commitments.",
    'CC': "Very high risk. Highly vulnerable, with default a real possibility.",
    'C': "Extremely high risk. Imminent default.",
    'D': "Default. Has defaulted on financial obligations."
}

# --- Plot Feature Contributions Function (from your original code) ---
def plot_feature_contributions(model, features, model_name):
    """
    Plots feature contributions (importance) for a given model.
    """
    try:
        if isinstance(model, CatBoostClassifier):
            feature_importances = model.get_feature_importance()
            sorted_idx = np.argsort(feature_importances)[::-1]
            feature_names = features
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(np.array(feature_names)[sorted_idx], feature_importances[sorted_idx], color='#3498db')
            ax.set_xlabel("Feature Importance", fontsize=12)
            ax.set_title(f"Feature Importance ({model_name})", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        elif isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, lgb.Booster)):
            if isinstance(model, lgb.Booster):
                feature_importances = model.feature_importance()
            else:
                feature_importances = model.feature_importances_
            
            sorted_idx = np.argsort(feature_importances)[::-1]
            feature_names = features
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(np.array(feature_names)[sorted_idx], feature_importances[sorted_idx], color='#3498db')
            ax.set_xlabel("Feature Importance", fontsize=12)
            ax.set_title(f"Feature Importance ({model_name})", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Feature importance plotting is not supported for this model type.")
    except Exception as e:
        st.error(f"Error plotting feature contributions: {e}")

# --- Helper function for resetting the app ---
def reset_inputs():
    """Resets all session state variables."""
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()


# --- Main App Logic ---
st.header("Credit Rating & Sentiment Predictor")
st.markdown("A tool to predict credit ratings based on financial metrics and sentiment from news articles.")

# Use Streamlit tabs for a cleaner, organized layout
tab1, tab2 = st.tabs(["Financial & Sentiment Input", "Prediction Results"])

# --- Tab 1: Input Form ---
with tab1:
    st.subheader("Financial Metrics & Sentiment Analysis")
    
    # Model Selection
    model_name_to_run = st.selectbox(
        "**Select a Model**",
        list(models.keys()),
        key="selected_model",
        help="Choose a model trained with either financial data only, or financial and sentiment data."
    )
    
    # Differentiate between financial-only and financial+sentiment models
    is_sentiment_model = "Sentiment" in model_name_to_run
    
    # Create an empty dictionary to hold user inputs
    input_features = {}

    st.markdown("---")
    
    st.subheader("Financial Metrics")
    
    # Use expanders to group financial inputs by category
    for category, cols in FINANCIAL_CATEGORIES.items():
        with st.expander(f"**{category}**", expanded=category == "Liquidity Ratios"):
            num_cols = len(cols)
            # Use columns for a compact layout
            num_cols_display = 3
            chunks = [cols[i:i + num_cols_display] for i in range(0, len(cols), num_cols_display)]
            
            for chunk in chunks:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if len(chunk) > 0:
                        metric_name = chunk[0]
                        min_val, max_val = FINANCIAL_INPUT_BOUNDS.get(metric_name, (None, None))
                        input_features[metric_name] = st.number_input(
                            f"Enter {metric_name}", 
                            value=0.0, 
                            step=0.01, 
                            format="%.2f", 
                            key=metric_name,
                            min_value=min_val,
                            max_value=max_val
                        )
                with col2:
                    if len(chunk) > 1:
                        metric_name = chunk[1]
                        min_val, max_val = FINANCIAL_INPUT_BOUNDS.get(metric_name, (None, None))
                        input_features[metric_name] = st.number_input(
                            f"Enter {metric_name}", 
                            value=0.0, 
                            step=0.01, 
                            format="%.2f", 
                            key=metric_name,
                            min_value=min_val,
                            max_value=max_val
                        )
                with col3:
                    if len(chunk) > 2:
                        metric_name = chunk[2]
                        min_val, max_val = FINANCIAL_INPUT_BOUNDS.get(metric_name, (None, None))
                        input_features[metric_name] = st.number_input(
                            f"Enter {metric_name}", 
                            value=0.0, 
                            step=0.01, 
                            format="%.2f", 
                            key=metric_name,
                            min_value=min_val,
                            max_value=max_val
                        )

    st.markdown("---")

    # --- Sentiment Analysis Section ---
    if is_sentiment_model:
        st.subheader("Sentiment Analysis from URLs")
        st.info("The selected model uses sentiment data. Paste URLs to news articles or reports to analyze.")
        
        # Initialize session state for URLs
        if 'urls' not in st.session_state:
            st.session_state.urls = [""] * 3
            st.session_state.sentiment_results = None

        url_list = []
        for i in range(3):
            url_list.append(st.text_input(f"URL {i+1}", st.session_state.urls[i], key=f"url_{i}"))

        if st.button("Analyze Sentiment"):
            with st.spinner("Analyzing sentiment... this may take a moment."):
                successful_urls = [url for url in url_list if url.strip()]
                if not successful_urls:
                    st.session_state.sentiment_results = {'Avg_Positive': 0.0, 'Avg_Neutral': 1.0, 'Avg_Negative': 0.0, 'Avg_Compound': 0.0, 'category': 'Neutral'}
                    st.warning("Please enter at least one URL to analyze sentiment.")
                else:
                    overall_sentiment, _ = analyze_multiple_urls_sentiment(successful_urls)
                    st.session_state.sentiment_results = overall_sentiment
            if st.session_state.sentiment_results:
                st.success("Sentiment analysis complete!")
                st.write(f"**Overall Sentiment:** {st.session_state.sentiment_results['category']}")
                sentiment_df = pd.DataFrame([st.session_state.sentiment_results]).T
                sentiment_df.columns = ["Score"]
                st.dataframe(sentiment_df.style.format("{:.2f}").set_properties(**{'border-radius': '8px'}), use_container_width=True)

    else:
        st.info("The selected model does not use sentiment data.")
        st.session_state.sentiment_results = {'Avg_Positive': 0.0, 'Avg_Neutral': 1.0, 'Avg_Negative': 0.0, 'Avg_Compound': 0.0, 'category': 'Neutral'}


    st.markdown("---")
    
    # Predict button
    if st.button("Predict Credit Rating", type="primary"):
        st.session_state['predict_clicked'] = True
        # Rerun to switch to the results tab and show results
        st.experimental_rerun()

# --- Tab 2: Results Display ---
with tab2:
    if 'predict_clicked' in st.session_state and st.session_state['predict_clicked']:
        st.subheader("Prediction Results")
        
        with st.spinner("Making prediction..."):
            
            # Combine financial and sentiment inputs based on the model
            input_data = {**input_features, **st.session_state.sentiment_results}
            
            # Select the correct feature list for the model
            features_for_model_call = ALL_COLS if "Sentiment" in model_name_to_run else FINANCIAL_COLS
            
            # Ensure the input data has all the required features for the model
            features_to_predict = {k: input_data.get(k, 0.0) for k in features_for_model_call}
            input_df_for_prediction = pd.DataFrame([features_to_predict])
            
            # Perform prediction
            model = models[model_name_to_run]
            scaler = scalers[model_name_to_run]
            predicted_rating, probabilities = _predict_single_model(
                model, scaler, input_df_for_prediction, features_for_model_call, label_encoder
            )
            
            if predicted_rating and predicted_rating != "Prediction failed.":
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Predicted Rating", value=predicted_rating)
                
                with col2:
                    st.markdown("### Definition")
                    st.info(f"**{predicted_rating}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating, 'Definition not available.')}")

                st.markdown("---")

                st.subheader("Probability Distribution")
                
                # Sort probabilities by the predefined RATING_ORDER
                prob_df = pd.DataFrame(probabilities.items(), columns=['Rating', 'Probability'])
                prob_df['Rating'] = pd.Categorical(prob_df['Rating'], categories=RATING_ORDER, ordered=True)
                prob_df = prob_df.sort_values('Rating')
                
                # Use a bar chart to visualize probabilities
                st.bar_chart(prob_df.set_index('Rating'))
                
                st.markdown("---")
                
                with st.expander("**Model Insights: Feature Importance**", expanded=False):
                    plot_feature_contributions(
                        model,
                        features_for_model_call, # Pass the correct features
                        model_name_to_run
                    )
            else:
                st.error("Prediction failed. Please check your inputs and try again.")
    else:
        st.info("Please fill out the financial metrics and sentiment URLs in the 'Financial & Sentiment Input' tab and click 'Predict'.")

# --- Reset Button (placed at the bottom for accessibility) ---
st.markdown("---")
st.button("Reset All Inputs", on_click=reset_inputs)

st.markdown("---")
st.info("Developed with Streamlit by your AI assistant.")








