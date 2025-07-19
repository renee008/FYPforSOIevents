import streamlit as st
import pandas as pd
import joblib # For loading the scalers and models
from catboost import CatBoostClassifier
from textblob import TextBlob # For basic sentiment analysis

# --- Configuration ---
st.set_page_config(page_title="Credit Rating & Sentiment Predictor", page_icon="📈", layout="centered")

# --- Define Feature Columns (MUST match your training script) ---
financial_cols = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin',
    'returnOnAssets', 'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover',
    'fixedAssetTurnover', 'debtEquityRatio', 'debtRatio', 'effectiveTaxRate',
    'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare',
    'companyEquityMultiplier', 'ebitPerRevenue', 'enterpriseValueMultiple',
    'payablesTurnover','operatingCashFlowPerShare', 'operatingCashFlowSalesRatio'
]

sentiment_cols = ['Avg_Positive', 'Avg_Neutral', 'Avg_Negative', 'Avg_Compound']
all_cols = financial_cols + sentiment_cols

# --- Model and Scaler Loading ---
# IMPORTANT: Ensure these files are in the same directory as this script on GitHub.

@st.cache_resource # Cache the models and scalers to avoid reloading on every rerun
def load_models_and_scalers():
    models = {}
    scalers = {}

    # Load Model A (Financial Only) and its scaler
    try:
        model_A = CatBoostClassifier()
        model_A.load_model('CatboostML.modelA.cbm')
        models['A'] = model_A
        st.success("Model A (Financial Only) loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Model A: {e}")
        st.warning("Please ensure 'CatboostML.modelA.cbm' is in the same directory.")

    try:
        scaler_fin = joblib.load('scaler_fin.pkl')
        scalers['fin'] = scaler_fin
        st.success("Scaler for Model A loaded successfully!")
    except Exception as e:
        st.error(f"Error loading scaler_fin.pkl: {e}")
        st.warning("Please ensure 'scaler_fin.pkl' is in the same directory.")

    # Load Model B (Financial + Sentiment) and its scaler
    try:
        model_B = CatBoostClassifier()
        model_B.load_model('CatboostML.modelB.cbm')
        models['B'] = model_B
        st.success("Model B (Financial + Sentiment) loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Model B: {e}")
        st.warning("Please ensure 'CatboostML.modelB.cbm' is in the same directory.")

    try:
        scaler_all = joblib.load('scaler_all.pkl')
        scalers['all'] = scaler_all
        st.success("Scaler for Model B loaded successfully!")
    except Exception as e:
        st.error(f"Error loading scaler_all.pkl: {e}")
        st.warning("Please ensure 'scaler_all.pkl' is in the same directory.")

    return models, scalers

models, scalers = load_models_and_scalers()

# --- Prediction Functions ---

def predict_credit_rating(model, scaler, input_df, feature_columns) -> str:
    """
    Predicts the credit rating using the given model and scaler.
    """
    if model is None or scaler is None:
        return "Model or scaler not loaded. Cannot predict credit rating."

    try:
        # Ensure input_df has the correct columns in the correct order
        # Reindex to ensure consistency with training features
        input_df_reindexed = input_df[feature_columns]
        
        # Scale the input features
        scaled_data = scaler.transform(input_df_reindexed)
        
        # Make prediction
        predicted_rating = model.predict(scaled_data)[0]
        
        return str(predicted_rating)

    except Exception as e:
        st.error(f"Error during credit rating prediction: {e}")
        return "Prediction failed."

def analyze_sentiment(news_article: str) -> dict:
    """
    Analyzes the sentiment of a news article and returns polarity and subjectivity.
    """
    if not news_article:
        return {'polarity': 0.0, 'subjectivity': 0.0, 'category': 'Neutral'}

    try:
        analysis = TextBlob(news_article)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Determine sentiment category based on polarity
        if polarity > 0.1: # Positive threshold
            category = "Positive"
        elif polarity < -0.1: # Negative threshold
            category = "Negative"
        else:
            category = "Neutral"
            
        return {'polarity': polarity, 'subjectivity': subjectivity, 'category': category}
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.0, 'category': 'Error'}

# --- Streamlit UI ---

st.title("Company Financial Health & News Sentiment Analyzer")
st.markdown("""
This application predicts a company's credit rating using two models:
1.  **Model A**: Based on financial metrics only.
2.  **Model B**: Based on financial metrics and news sentiment.
""")

# --- Input Fields for Financial Metrics (Common to both models) ---
st.header("Enter Financial Metrics")

company_name = st.text_input("Company Name", "Example Corp")

# Create a dictionary to hold all financial inputs
financial_inputs = {}
for col in financial_cols:
    # Provide sensible default values and ranges based on typical financial ratios
    # You might need to adjust these based on your actual data's distribution
    default_value = 0.0
    min_val = -1000.0 # Allow for negative values in some metrics like Return on Assets
    max_val = 1000.0
    step_val = 0.01

    if 'Ratio' in col or 'Multiplier' in col or 'Turnover' in col:
        default_value = 1.0
        min_val = 0.0
        max_val = 10.0
        step_val = 0.01
    elif 'Margin' in col or 'ReturnOn' in col or 'TaxRate' in col:
        # These are often percentages, convert to decimal for model input
        default_value = 0.10 # 10%
        min_val = -1.0 # -100%
        max_val = 1.0 # 100%
        step_val = 0.001
        st.write(f"**{col} (as decimal, e.g., 0.1 for 10%)**")
        financial_inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_value, step=step_val, key=f"fin_{col}")
    else:
        financial_inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_value, step=step_val, key=f"fin_{col}")

# Convert financial inputs to a DataFrame row
financial_df_row = pd.DataFrame([financial_inputs])


st.markdown("---")

# --- Model A Prediction Section (Financial Only) ---
st.header("1. Predict Credit Rating (Model A - Financial Only)")
st.caption("This model uses only the financial metrics entered above.")

if st.button("Predict with Model A"):
    if 'A' in models and 'fin' in scalers:
        with st.spinner("Predicting with Model A..."):
            predicted_rating_A = predict_credit_rating(models['A'], scalers['fin'], financial_df_row, financial_cols)
            st.subheader(f"Predicted Credit Rating (Model A) for {company_name}:")
            st.success(f"**{predicted_rating_A}**")
    else:
        st.warning("Model A or its scaler not loaded. Cannot perform prediction.")

st.markdown("---")

# --- Model B Prediction Section (Financial + Sentiment) ---
st.header("2. Predict Credit Rating (Model B - Financial + Sentiment)")
st.caption("This model combines financial metrics with news article sentiment.")

news_article = st.text_area("Enter Company News Article Here",
                            "Example: The company announced record profits this quarter, exceeding all expectations and leading to a significant stock price increase. However, concerns about market competition are rising.",
                            height=200)

if st.button("Analyze Sentiment & Predict with Model B"):
    sentiment_result = analyze_sentiment(news_article)
    
    st.subheader("News Article Sentiment:")
    if sentiment_result['category'] == "Positive":
        st.success(f"Sentiment: **{sentiment_result['category']}** 😊 (Polarity: {sentiment_result['polarity']:.2f}, Subjectivity: {sentiment_result['subjectivity']:.2f})")
    elif sentiment_result['category'] == "Negative":
        st.error(f"Sentiment: **{sentiment_result['category']}** 😠 (Polarity: {sentiment_result['polarity']:.2f}, Subjectivity: {sentiment_result['subjectivity']:.2f})")
    else:
        st.info(f"Sentiment: **{sentiment_result['category']}** 😐 (Polarity: {sentiment_result['polarity']:.2f}, Subjectivity: {sentiment_result['subjectivity']:.2f})")

    # Prepare data for Model B prediction
    if 'B' in models and 'all' in scalers:
        sentiment_data = {
            'Avg_Positive': [1 if sentiment_result['polarity'] > 0.1 else 0], # Simple binary for demo
            'Avg_Neutral': [1 if -0.1 <= sentiment_result['polarity'] <= 0.1 else 0],
            'Avg_Negative': [1 if sentiment_result['polarity'] < -0.1 else 0],
            'Avg_Compound': [sentiment_result['polarity']] # Using polarity as compound score
        }
        # Create a DataFrame for sentiment features
        sentiment_df_row = pd.DataFrame([sentiment_data])

        # Combine financial and sentiment data
        # Ensure column order matches 'all_cols'
        combined_df_row = pd.concat([financial_df_row, sentiment_df_row], axis=1)
        
        with st.spinner("Predicting with Model B..."):
            predicted_rating_B = predict_credit_rating(models['B'], scalers['all'], combined_df_row, all_cols)
            st.subheader(f"Predicted Credit Rating (Model B) for {company_name}:")
            st.success(f"**{predicted_rating_B}**")
    else:
        st.warning("Model B or its scaler not loaded. Cannot perform prediction.")

st.markdown("---")
st.info("Developed with Streamlit by your AI assistant. Remember to train and save your models and scalers!")

