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

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        padding: 8px 12px;
    }
    .st-emotion-cache-nahz7x { /* Target Streamlit tabs container */
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 20px;
    }
    .st-emotion-cache-10q7065 { /* Target individual tabs */
        border-radius: 12px 12px 0 0;
    }
    .st-emotion-cache-1r6dm1x { /* Target expander/popover */
        border-radius: 12px;
    }
    /* --- CSS to hide number input arrows --- */
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type=number] {
        -moz-appearance: textfield; /* Firefox */
    }

    /* --- Custom CSS for Selectbox (Dropdown) --- */
    /* Target the container of the selectbox */
    .st-emotion-cache-1n76tmc { /* This class might change slightly with Streamlit updates */
        background-color: #e8f5e9; /* Light green background */
        border-radius: 12px;
        border: 1px solid #a5d6a7; /* Green border */
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    /* Target the actual dropdown button */
    .st-emotion-cache-1n76tmc > div > div > button {
        background-color: #c8e6c9; /* Slightly darker green for the button */
        color: #1b5e20; /* Dark green text */
        font-weight: 700;
        border-radius: 10px;
        border: none;
    }
    /* Target the options list when opened */
    .st-emotion-cache-1n76tmc .st-emotion-cache-1x0x577 { /* This targets the options container */
        background-color: #f1f8e9; /* Very light green for options background */
        border-radius: 8px;
        border: 1px solid #a5d6a7;
    }
    /* Target individual options */
    .st-emotion-cache-1n76tmc .st-emotion-cache-1x0x577 div {
        color: #333; /* Dark text for options */
    }
    /* Hover state for options */
    .st-emotion-cache-1n76tmc .st-emotion-cache-1x0x577 div:hover {
        background-color: #dcedc8; /* Lighter green on hover */
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
st.set_page_config(page_title="Credit Rating & Sentiment Predictor", page_icon="📈", layout="wide") # Changed to wide layout

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
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text from common content tags
            paragraphs = soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'li'])
            text = " ".join([p.get_text(separator=' ', strip=True) for p in paragraphs])
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()

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
    'A': "High quality, low risk. Strong capacity to meet financial commitments, but somewhat more susceptible to adverse economic conditions.",
    'BBB': "Good quality, moderate risk. Adequate capacity to meet financial commitments, but more susceptible to adverse economic conditions than higher-rated categories.",
    'BB': "Speculative, significant risk. Less vulnerable in the near term but faces major uncertainties and exposure to adverse business, financial, or economic conditions.",
    'B': "Highly speculative, high risk. Significant credit risk, with a limited margin of safety.",
    'CCC': "Substantial credit risk. Currently vulnerable, and dependent upon favorable business, financial, and economic conditions to meet financial commitments.",
    'CC': "Very high credit risk. Highly vulnerable, with default a strong possibility.",
    'C': "Extremely high credit risk. Default is imminent or has occurred, with little prospect for recovery.",
    'D': "Default. The company has defaulted on its financial obligations."
}

# --- Feature Importance Plotting Function ---
def plot_feature_contributions(model, feature_columns, model_label):
    """
    Calculates and plots global feature importances for the given model.
    The importances are scaled such that the most important feature has a value of 10.
    The X-axis is set to 0-10.
    """
    try:
        feature_importances = None
        
        if isinstance(model, CatBoostClassifier):
            feature_importances = model.get_feature_importance()
        elif isinstance(model, RandomForestClassifier) or isinstance(model, xgb.XGBClassifier):
            feature_importances = model.feature_importances_
        elif isinstance(model, lgb.Booster):
            feature_importances = model.feature_importance()
        else:
            st.warning(f"Feature importance plotting not supported for model type: {type(model).__name__}.")
            return

        if feature_importances is None or len(feature_importances) == 0:
            st.info(f"Could not retrieve feature importances for {model_label}.")
            return

        # Scale the importances such that the maximum value becomes 10
        max_importance = np.max(feature_importances)
        if max_importance > 0:
            scaled_importances = (feature_importances / max_importance) * 10
        else:
            scaled_importances = np.zeros_like(feature_importances) # All zeros if max is zero


        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': scaled_importances # Use scaled importances for plotting
        })
        
        # Sort by importance for better visualization
        importance_df = importance_df.sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(5, len(feature_columns) * 0.3))) # Dynamic height, slightly smaller for columns
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#4CAF50') # Green bars
        ax.set_xlabel("Feature Importance Score (Scaled to Max 10)") # Updated label
        ax.set_title(f"Overall Feature Contributions:\n{model_label}", fontsize=10) # Smaller title for columns
        
        # Set X-axis limits to 0-10
        ax.set_xlim(0, 10)
        
        # Set X-axis ticks and labels to 0-10 (integers)
        ax.set_xticks(np.arange(0, 11, 1)) # Ticks from 0 to 10, step 1
        ax.set_xticklabels([str(int(x)) for x in np.arange(0, 11, 1)]) # Labels as integers

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        st.error(f"Could not generate Feature Importance plot for {model_label}: {e}. Ensure the model object has a 'feature_importances_' attribute or a 'get_feature_importance()' method.")


# --- Default Values and Ranges for Inputs ---
default_values = {
    'currentRatio': 1.8, 'quickRatio': 1.0, 'cashRatio': 0.25, 'daysOfSalesOutstanding': 35.0,
    'netProfitMargin': 0.10, 'pretaxProfitMargin': 0.12, 'grossProfitMargin': 0.30,
    'returnOnAssets': 0.08, 'returnOnEquity': 0.20, 'assetTurnover': 1.2,
    'fixedAssetTurnover': 3.5, 'debtRatio': 0.4, 'effectiveTaxRate': 0.28,
    'freeCashFlowOperatingCashFlowRatio': 0.75, 'freeCashFlowPerShare': 2.5, 'cashPerShare': 3.0,
    'enterpriseValueMultiple': 12.0, 'operatingCashFlowPerShare': 3.0, 'operatingCashFlowSalesRatio': 0.12, 'payablesTurnover': 9.0
}

min_values = {
    'currentRatio': 0.0, 'quickRatio': 0.0, 'cashRatio': 0.0, 'daysOfSalesOutstanding': 0.0,
    'netProfitMargin': -5.0, 'pretaxProfitMargin': -5.0, 'grossProfitMargin': -5.0,
    'returnOnAssets': -5.0, 'returnOnEquity': -5.0, 'assetTurnover': 0.0,
    'fixedAssetTurnover': 0.0, 'debtRatio': 0.0, 'effectiveTaxRate': 0.0,
    'freeCashFlowOperatingCashFlowRatio': -10.0, 'freeCashFlowPerShare': -100.0, 'cashPerShare': 0.0,
    'enterpriseValueMultiple': 0.0,
    'operatingCashFlowPerShare': -100.0, 'operatingCashFlowSalesRatio': -5.0, 'payablesTurnover': 0.0
}

max_values = {
    'currentRatio': 10.0, 'quickRatio': 5.0, 'cashRatio': 1.0, 'daysOfSalesOutstanding': 365.0,
    'netProfitMargin': 1.0, 'pretaxProfitMargin': 1.0, 'grossProfitMargin': 1.0,
    'returnOnAssets': 1.0, 'returnOnEquity': 1.0, 'assetTurnover': 5.0,
    'fixedAssetTurnover': 20.0, 'debtRatio': 1.0, 'effectiveTaxRate': 1.0,
    'freeCashFlowOperatingCashFlowRatio': 5.0, 'freeCashFlowPerShare': 100.0, 'cashPerShare': 50.0,
    'enterpriseValueMultiple': 50.0,
    'operatingCashFlowPerShare': 100.0, 'operatingCashFlowSalesRatio': 1.0, 'payablesTurnover': 20.0
}

step_values = {
    'currentRatio': 0.01, 'quickRatio': 0.01, 'cashRatio': 0.01, 'daysOfSalesOutstanding': 1.0,
    'netProfitMargin': 0.001, 'pretaxProfitMargin': 0.001, 'grossProfitMargin': 0.001,
    'returnOnAssets': 0.001, 'returnOnEquity': 0.001, 'assetTurnover': 0.01,
    'fixedAssetTurnover': 0.01, 'debtRatio': 0.001, 'effectiveTaxRate': 0.001,
    'freeCashFlowOperatingCashFlowRatio': 0.01, 'freeCashFlowPerShare': 0.1, 'cashPerShare': 0.1,
    'enterpriseValueMultiple': 0.1,
    'operatingCashFlowPerShare': 0.1, 'operatingCashFlowSalesRatio': 0.001, 'payablesTurnover': 0.01
}


# --- Initialize Session State for Reset Button and Prediction Triggers ---
if 'financial_inputs' not in st.session_state:
    st.session_state.financial_inputs = {col: default_values.get(col, 0.0) for col in FINANCIAL_COLS}
if 'news_article_urls' not in st.session_state: # Changed to store multiple URLs
    st.session_state.news_article_urls = "" # Initialize as blank
if 'company_name' not in st.session_state:
    st.session_state.company_name = "Example Corp"
if 'last_extracted_texts' not in st.session_state: # To store extracted texts for display
    st.session_state.last_extracted_texts = {}


def reset_inputs():
    st.session_state.financial_inputs = {col: default_values.get(col, 0.0) for col in FINANCIAL_COLS}
    st.session_state.news_article_urls = "" # Reset to blank
    st.session_state.company_name = "Example Corp"
    st.session_state.last_extracted_texts = {}


# --- Main UI Layout ---
st.title("Company Financial Health & News Sentiment Analyzer")

# Navigation Bar (using st.tabs)
tab_about, tab_how_to_use = st.tabs(["About Credit Ratings", "How to Use This Website"])

with tab_about:
    st.markdown("""
    ### What is a Credit Rating?
    A credit rating is an independent assessment of a company's financial strength and its ability to meet its financial obligations. These ratings are crucial for investors, lenders, and businesses as they provide a quick snapshot of creditworthiness, influencing borrowing costs and investment decisions. Ratings typically range from 'AAA' (highest quality, lowest risk) to 'D' (default).

    ### How are Credit Ratings Calculated?
    Credit rating agencies use a comprehensive approach, combining quantitative financial analysis with qualitative factors.
    * **Financial Metrics (Quantitative):** This involves analyzing a company's balance sheet, income statement, and cash flow statement. Key ratios like liquidity (e.g., current ratio), profitability (e.g., gross profit margin), leverage (e.g., debt ratio), and efficiency (e.g., days of sales outstanding) are vital.
    * **News Sentiment (Qualitative):** Public sentiment and news coverage can significantly impact a company's perceived risk. Positive news might signal stability and growth, while negative news could indicate potential challenges.

    This application uses eight machine learning models, grouped into 'Model A' (Financial Only) and 'Model B' (Financial + Sentiment):
    * **CatBoost Model A (Financial Only):** Predicts credit ratings based solely on a set of key financial metrics using a CatBoost Classifier.
    * **CatBoost Model B (Financial + Sentiment):** Combines financial metrics with sentiment analysis from news articles using a CatBoost Classifier.
    * **RandomForest Model A (Financial Only):** Predicts credit ratings based solely on a set of key financial metrics using a RandomForest Classifier.
    * **RandomForest Model B (Financial + Sentiment):** Combines financial metrics with sentiment analysis from news articles using a RandomForest Classifier.
    * **LightGBM Model A (Financial Only):** Predicts credit ratings based solely on a set of key financial metrics using a LightGBM Classifier.
    * **LightGBM Model B (Financial + Sentiment):** Combines financial metrics with sentiment analysis from news articles using a LightGBM Classifier.
    * **XGBoost Model A (Financial Only):** Predicts credit ratings based solely on a set of key financial metrics using an XGBoost Classifier.
    * **XGBoost Model B (Financial + Sentiment):** Combines financial metrics with sentiment analysis from news articles using an XGBoost Classifier.
    """)

with tab_how_to_use:
    st.markdown("""
    ### How to Use This Website
    1.  **Enter Company Name:** Provide the name of the company you are analyzing.
    2.  **Select a Model:** Choose one of the individual models, or an "All Models" option, from the dropdown.
    3.  **Input Financial Metrics:** Fill in the values for the various financial ratios. Enter percentage-like metrics (e.g., margins, returns) as decimals (e.g., `0.10` for 10%).
    4.  **Enter News Article URL(s) (for Model B types):** If you select any "Model B" option (individual or "All Model B"), provide one or more relevant news article URLs, one per line. The app will attempt to scrape and analyze the content and average the sentiment.
    5.  **Predict:** Click the "Predict Credit Rating" button to get a prediction and feature contributions.
    6.  **Review Results:** The predicted rating(s), probabilities, and key feature contributions will appear. For "All Models" options, results will be organized into side-by-side columns.
    7.  **Reset Inputs:** Use the "Reset All Inputs" button to clear the form and start fresh.
    """)

st.markdown("---") # Separator below the tabs

st.header("Enter Company Details")

st.session_state.company_name = st.text_input("Company Name", value=st.session_state.company_name, key="company_name_input")

st.markdown("---")

# --- Model Selection ---
st.header("Select Prediction Model")

# Define the model options for the selectbox
model_options = list(models.keys())
model_options.sort() # Sort individual models alphabetically
model_options_grouped = [
    "--- Individual Models ---"
] + model_options + [
    "--- Compare Models ---",
    "All Model A (Financial Only)",
    "All Model B (Financial + Sentiment)",
    "All Models (A & B)"
]

selected_model_name = st.selectbox(
    "Choose a model for prediction:",
    model_options_grouped,
    key="model_selector"
)

# Determine which features are needed based on the selected option
sentiment_input_needed = False
required_features = [] # Initialize with empty list

if selected_model_name in models: # An individual model is selected
    selected_model = models[selected_model_name]
    selected_scaler = scalers[selected_model_name]
    if "Financial Only" in selected_model_name:
        required_features = FINANCIAL_COLS
        st.info(f"You selected '{selected_model_name}'. This model uses only financial metrics.")
    else: # Implies "Financial + Sentiment"
        required_features = ALL_COLS
        st.info(f"You selected '{selected_model_name}'. This model uses both financial metrics and news sentiment.")
        sentiment_input_needed = True
elif selected_model_name == "All Model A (Financial Only)":
    required_features = FINANCIAL_COLS
    st.info("You selected 'All Model A'. This will run all four 'Financial Only' models and display them side-by-side.")
    sentiment_input_needed = False
elif selected_model_name == "All Model B (Financial + Sentiment)":
    required_features = ALL_COLS
    st.info("You selected 'All Model B'. This will run all four 'Financial + Sentiment' models and display them side-by-side.")
    sentiment_input_needed = True
elif selected_model_name == "All Models (A & B)":
    required_features = ALL_COLS # All models use all features eventually for Model B types
    st.info("You selected 'All Models (A & B)'. This will run all eight models and display them side-by-side.")
    sentiment_input_needed = True
else: # Placeholder or separator selected
    selected_model = None
    selected_scaler = None
    st.warning("Please select a valid model or comparison option.")


st.markdown("---")

# --- Input Fields for Financial Metrics (Enhanced UI) ---
st.header("Enter Financial Metrics")
st.markdown("*(All values should be numerical. Enter percentages as decimals, e.g., 0.1 for 10%)*")

# Iterate through categories and create expanders for each
for category, cols_in_category in FINANCIAL_CATEGORIES.items():
    with st.expander(f"**{category}**"):
        num_cols_per_row_in_expander = 2
        expander_cols = st.columns(num_cols_per_row_in_expander)
        
        for i, col_name in enumerate(cols_in_category):
            with expander_cols[i % num_cols_per_row_in_expander]:
                st.write(f"**{col_name}**")
                if 'Margin' in col_name or 'ReturnOn' in col_name or 'TaxRate' in col_name or 'Ratio' in col_name:
                    st.caption("Enter as decimal (e.g., 0.1 for 10%)")
                
                metric_help_text = {
                    'currentRatio': 'Measures short-term liquidity: current assets / current liabilities.',
                    'quickRatio': 'Measures short-term liquidity excluding inventory: (current assets - inventory) / current liabilities.',
                    'cashRatio': 'Most conservative liquidity measure: cash / current liabilities.',
                    'daysOfSalesOutstanding': 'Average number of days to collect accounts receivable.',
                    'netProfitMargin': 'Percentage of revenue left after all expenses, including taxes.',
                    'pretaxProfitMargin': 'Percentage of revenue left before taxes.',
                    'grossProfitMargin': 'Percentage of revenue left after deducting cost of goods sold.',
                    'returnOnAssets': 'How efficiently a company uses its assets to generate earnings.',
                    'returnOnEquity': 'How much profit a company generates for each dollar of shareholders\' equity.',
                    'assetTurnover': 'How efficiently a company uses its assets to generate sales.',
                    'fixedAssetTurnover': 'How efficiently a company uses its fixed assets to generate sales.',
                    'debtRatio': 'Total debt / total assets. Measures leverage.',
                    'effectiveTaxRate': 'The actual rate of tax paid by a company on its earnings.',
                    'freeCashFlowOperatingCashFlowRatio': 'Free cash flow / operating cash flow. Measures cash available after capital expenditures.',
                    'freeCashFlowPerShare': 'Free cash flow available per share.',
                    'cashPerShare': 'Cash and cash equivalents per outstanding share.',
                    'enterpriseValueMultiple': 'Enterprise Value / EBITDA. Valuation multiple.',
                    'operatingCashFlowPerShare': 'Cash generated from operations per share.',
                    'operatingCashFlowSalesRatio': 'Operating cash flow / sales. Measures cash generated from each dollar of sales.',
                    'payablesTurnover': 'How many times a company pays off its accounts payable during a period.'
                }
                st.info(metric_help_text.get(col_name, "No specific help text available."), icon="ℹ️")

                st.session_state.financial_inputs[col_name] = st.number_input(
                    f"Value for {col_name}",
                    min_value=min_values.get(col_name, 0.0),
                    max_value=max_values.get(col_name, 1000.0),
                    value=st.session_state.financial_inputs.get(col_name, default_values.get(col_name, 0.0)),
                    step=step_values.get(col_name, 0.01),
                    key=f"fin_input_{col_name}" # Unique key for each input
                )

# Convert financial inputs to a DataFrame row
financial_df_row = pd.DataFrame([st.session_state.financial_inputs])

st.markdown("---")

# --- Sentiment Input (URL-based, multiple URLs) ---
overall_sentiment_result = None
if sentiment_input_needed:
    st.header("Enter News Article URL(s) (for Sentiment Analysis)")
    st.session_state.news_article_urls = st.text_area(
        "Paste News Article URL(s) Here (one per line)",
        value=st.session_state.news_article_urls,
        height=150,
        key="news_article_urls_input"
    )

    urls_list = [url.strip() for url in st.session_state.news_article_urls.split('\n') if url.strip()]

    if urls_list:
        with st.spinner("Fetching and analyzing articles from URLs... This may take a moment for multiple links."):
            overall_sentiment_result, detailed_url_results = analyze_multiple_urls_sentiment(urls_list)
        
        if overall_sentiment_result and 'category' in overall_sentiment_result:
            st.subheader("Overall News Article Sentiment (Average):")
            st.info(f"VADER Compound Score: {overall_sentiment_result['Avg_Compound']:.2f} (Positive: {overall_sentiment_result['Avg_Positive']:.2f}, Neutral: {overall_sentiment_result['Avg_Neutral']:.2f}, Negative: {overall_sentiment_result['Avg_Negative']:.2f})")

            if overall_sentiment_result['category'] == "Positive":
                st.success(f"Overall Sentiment Category: **{overall_sentiment_result['category']}** 😊")
            elif overall_sentiment_result['category'] == "Negative":
                st.error(f"Overall Sentiment Category: **{overall_sentiment_result['category']}** 😠")
            else:
                st.info(f"Overall Sentiment Category: **{overall_sentiment_result['category']}** 😐")
            
            with st.expander("View Detailed URL Analysis Results"):
                for url, result in detailed_url_results.items():
                    st.markdown(f"**URL:** [{url}]({url})")
                    if result["status"] == "Success":
                        # Removed 'Category: {result['sentiment']['category']}'
                        st.success(f"  Status: Success (Compound: {result['sentiment']['Avg_Compound']:.2f})")
                    else:
                        st.error(f"  Status: Failed - {result['error']}")
                    st.markdown("---")
        else:
            st.error("Could not perform overall sentiment analysis. Please check the URLs.")
            overall_sentiment_result = {'Avg_Positive': 0.0, 'Avg_Neutral': 1.0, 'Avg_Negative': 0.0, 'Avg_Compound': 0.0, 'category': 'Neutral'} # Default neutral to avoid errors downstream
    else:
        st.info("Enter one or more URLs above to analyze their sentiment.")
        overall_sentiment_result = {'Avg_Positive': 0.0, 'Avg_Neutral': 1.0, 'Avg_Negative': 0.0, 'Avg_Compound': 0.0, 'category': 'Neutral'} # Default neutral if no URL


st.markdown("---")

# --- Prediction Button ---
if st.button(f"Predict Credit Rating(s)", key="predict_button"):
    
    # Prepare input data based on selected model type
    current_financial_inputs_df = pd.DataFrame([st.session_state.financial_inputs])
    current_sentiment_inputs_dict = overall_sentiment_result # Use the overall sentiment result

    # Validate inputs before proceeding
    if not all(val is not None for val in st.session_state.financial_inputs.values()):
        st.warning("Please fill in all financial metrics.")
        st.stop()
    if sentiment_input_needed and (not urls_list or not overall_sentiment_result or 'category' not in overall_sentiment_result):
        st.warning("Please enter valid news article URL(s) and ensure overall sentiment analysis was successful.")
        st.stop()
    if selected_model_name.startswith("---"): # If a separator is selected
        st.warning("Please select a valid model or comparison option from the dropdown.")
        st.stop()

    st.header("3. Prediction Result(s)")

    if selected_model_name in models: # Individual model prediction
        input_df_for_prediction = current_financial_inputs_df.copy()
        if sentiment_input_needed:
            # Ensure only the relevant sentiment columns are passed
            sentiment_features_for_model = {col: current_sentiment_inputs_dict[col] for col in SENTIMENT_COLS}
            all_inputs = {**st.session_state.financial_inputs, **sentiment_features_for_model}
            input_df_for_prediction = pd.DataFrame([all_inputs])
        
        input_df_for_prediction = input_df_for_prediction[required_features] # Ensure correct order

        with st.spinner(f"Predicting with {selected_model_name}..."):
            predicted_rating, probabilities = _predict_single_model(selected_model, selected_scaler, input_df_for_prediction, required_features, label_encoder)
            
            st.success(f"The predicted credit rating for {st.session_state.company_name} using **{selected_model_name}** is: **{predicted_rating}**")

            with st.popover(f"What is '{predicted_rating}'?"):
                st.write(f"**{predicted_rating}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating, 'Definition not available.')}")

            st.write("---")
            st.subheader("Prediction Probabilities:")
            prob_df = pd.DataFrame(probabilities.items(), columns=['Rating', 'Probability'])
            prob_df['Probability'] = prob_df['Probability'].astype(float) # Ensure numeric for sorting
            prob_df = prob_df.sort_values(by='Probability', ascending=False)
            prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}") # Format as percentage
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

            st.write("---")
            st.subheader("Key Feature Contributions (Overall Importance):")
            
            plot_feature_contributions(
                selected_model,
                required_features,
                selected_model_name
            )

    else: # "All Models" options
        st.info("Running multiple models. Results will be displayed below.")
        
        # Filter models into A and B groups
        model_A_names = sorted([name for name in models.keys() if "Model A" in name])
        model_B_names = sorted([name for name in models.keys() if "Model B" in name])

        cols_per_row = 4

        # --- Display All Model A (Financial Only) ---
        if model_A_names: # Check if there are Model A models to display
            st.subheader("All Model A (Financial Only) Predictions:")
            # Use a container to group these columns for better visual separation
            with st.container():
                for i in range(0, len(model_A_names), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(model_A_names):
                            model_name_to_run = model_A_names[i + j]
                            model = models[model_name_to_run]
                            scaler = scalers[model_name_to_run]

                            features_for_model = FINANCIAL_COLS # Model A always uses financial features
                            input_df_for_prediction = current_financial_inputs_df.copy()
                            input_df_for_prediction = input_df_for_prediction[features_for_model]

                            with cols[j]:
                                st.markdown(f"**{model_name_to_run}**") # Use markdown for smaller title in column
                                with st.spinner(f"Predicting..."):
                                    predicted_rating, probabilities = _predict_single_model(model, scaler, input_df_for_prediction, features_for_model, label_encoder)
                                    
                                    if predicted_rating != "Prediction failed.":
                                        st.success(f"Rating: **{predicted_rating}**")
                                        with st.popover(f"What is '{predicted_rating}'?"):
                                            st.write(f"**{predicted_rating}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating, 'Definition not available.')}")

                                        st.markdown("**Probabilities:**")
                                        prob_df = pd.DataFrame(probabilities.items(), columns=['Rating', 'Probability'])
                                        prob_df['Probability'] = prob_df['Probability'].astype(float)
                                        prob_df = prob_df.sort_values(by='Probability', ascending=False)
                                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                                        st.dataframe(prob_df, hide_index=True, use_container_width=True)

                                        st.markdown("**Feature Importance:**")
                                        plot_feature_contributions(
                                            model,
                                            features_for_model,
                                            model_name_to_run
                                        )
                                    else:
                                        st.error(f"Prediction failed.")
            st.markdown("---") # Separator after Model A group

        # --- Display All Model B (Financial + Sentiment) ---
        if model_B_names: # Check if there are Model B models to display
            st.subheader("All Model B (Financial + Sentiment) Predictions:")
            with st.container(): # Use a container for visual separation
                for i in range(0, len(model_B_names), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(model_B_names):
                            model_name_to_run = model_B_names[i + j]
                            model = models[model_name_to_run]
                            scaler = scalers[model_name_to_run]

                            features_for_model = ALL_COLS # Model B always uses all features
                            # Ensure only the relevant sentiment columns are passed
                            sentiment_features_for_model = {col: current_sentiment_inputs_dict[col] for col in SENTIMENT_COLS}
                            all_inputs = {**st.session_state.financial_inputs, **sentiment_features_for_model}
                            input_df_for_prediction = pd.DataFrame([all_inputs])
                            input_df_for_prediction = input_df_for_prediction[features_for_model]

                            with cols[j]:
                                st.markdown(f"**{model_name_to_run}**") # Use markdown for smaller title in column
                                with st.spinner(f"Predicting..."):
                                    predicted_rating, probabilities = _predict_single_model(model, scaler, input_df_for_prediction, features_for_model, label_encoder)
                                    
                                    if predicted_rating != "Prediction failed.":
                                        st.success(f"Rating: **{predicted_rating}**")
                                        with st.popover(f"What is '{predicted_rating}'?"):
                                            st.write(f"**{predicted_rating}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating, 'Definition not available.')}")

                                        st.markdown("**Probabilities:**")
                                        prob_df = pd.DataFrame(probabilities.items(), columns=['Rating', 'Probability'])
                                        prob_df['Probability'] = prob_df['Probability'].astype(float)
                                        prob_df = prob_df.sort_values(by='Probability', ascending=False)
                                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                                        st.dataframe(prob_df, hide_index=True, use_container_width=True)

                                        st.markdown("**Feature Importance:**")
                                        plot_feature_contributions(
                                            model,
                                            features_for_model,
                                            model_name_to_run
                                        )
                                    else:
                                        st.error(f"Prediction failed.")


# --- Reset Button (placed at the bottom for accessibility) ---
st.markdown("---")
st.button("Reset All Inputs", on_click=reset_inputs)

st.markdown("---")
st.info("Developed with Streamlit by your AI assistant.")









