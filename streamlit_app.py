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

# Define sensible default values and ranges for each financial metric
# These are crucial for user experience and preventing invalid inputs
# Values are based on general financial health indicators.
default_values = {
    'currentRatio': 1.8, 'quickRatio': 1.0, 'cashRatio': 0.25, 'daysOfSalesOutstanding': 35.0,
    'netProfitMargin': 0.10, 'pretaxProfitMargin': 0.12, 'grossProfitMargin': 0.30, 'operatingProfitMargin': 0.15,
    'returnOnAssets': 0.08, 'returnOnCapitalEmployed': 0.15, 'returnOnEquity': 0.20, 'assetTurnover': 1.2,
    'fixedAssetTurnover': 3.5, 'debtEquityRatio': 0.7, 'debtRatio': 0.4, 'effectiveTaxRate': 0.28,
    'freeCashFlowOperatingCashFlowRatio': 0.75, 'freeCashFlowPerShare': 2.5, 'cashPerShare': 3.0,
    'companyEquityMultiplier': 2.5, 'ebitPerRevenue': 0.18, 'enterpriseValueMultiple': 12.0,
    'operatingCashFlowPerShare': 3.0, 'operatingCashFlowSalesRatio': 0.12, 'payablesTurnover': 9.0
}

min_values = {
    'currentRatio': 0.0, 'quickRatio': 0.0, 'cashRatio': 0.0, 'daysOfSalesOutstanding': 0.0,
    'netProfitMargin': -5.0, 'pretaxProfitMargin': -5.0, 'grossProfitMargin': -5.0, 'operatingProfitMargin': -5.0,
    'returnOnAssets': -5.0, 'returnOnCapitalEmployed': -5.0, 'returnOnEquity': -5.0, 'assetTurnover': 0.0,
    'fixedAssetTurnover': 0.0, 'debtEquityRatio': 0.0, 'debtRatio': 0.0, 'effectiveTaxRate': 0.0,
    'freeCashFlowOperatingCashFlowRatio': -10.0, 'freeCashFlowPerShare': -100.0, 'cashPerShare': 0.0,
    'companyEquityMultiplier': 1.0, 'ebitPerRevenue': -5.0, 'enterpriseValueMultiple': 0.0,
    'operatingCashFlowPerShare': -100.0, 'operatingCashFlowSalesRatio': -5.0, 'payablesTurnover': 0.0
}

max_values = {
    'currentRatio': 10.0, 'quickRatio': 5.0, 'cashRatio': 1.0, 'daysOfSalesOutstanding': 365.0,
    'netProfitMargin': 1.0, 'pretaxProfitMargin': 1.0, 'grossProfitMargin': 1.0, 'operatingProfitMargin': 1.0,
    'returnOnAssets': 1.0, 'returnOnCapitalEmployed': 1.0, 'returnOnEquity': 1.0, 'assetTurnover': 5.0,
    'fixedAssetTurnover': 20.0, 'debtEquityRatio': 10.0, 'debtRatio': 1.0, 'effectiveTaxRate': 1.0,
    'freeCashFlowOperatingCashFlowRatio': 5.0, 'freeCashFlowPerShare': 100.0, 'cashPerShare': 50.0,
    'companyEquityMultiplier': 10.0, 'ebitPerRevenue': 1.0, 'enterpriseValueMultiple': 50.0,
    'operatingCashFlowPerShare': 100.0, 'operatingCashFlowSalesRatio': 1.0, 'payablesTurnover': 20.0
}

step_values = {
    'currentRatio': 0.01, 'quickRatio': 0.01, 'cashRatio': 0.01, 'daysOfSalesOutstanding': 1.0,
    'netProfitMargin': 0.001, 'pretaxProfitMargin': 0.001, 'grossProfitMargin': 0.001, 'operatingProfitMargin': 0.001,
    'returnOnAssets': 0.001, 'returnOnCapitalEmployed': 0.001, 'returnOnEquity': 0.001, 'assetTurnover': 0.01,
    'fixedAssetTurnover': 0.01, 'debtEquityRatio': 0.01, 'debtRatio': 0.001, 'effectiveTaxRate': 0.001,
    'freeCashFlowOperatingCashFlowRatio': 0.01, 'freeCashFlowPerShare': 0.1, 'cashPerShare': 0.1,
    'companyEquityMultiplier': 0.01, 'ebitPerRevenue': 0.001, 'enterpriseValueMultiple': 0.1,
    'operatingCashFlowPerShare': 0.1, 'operatingCashFlowSalesRatio': 0.001, 'payablesTurnover': 0.01
}


for col in financial_cols:
    st.write(f"**{col}**")
    # Add a note for percentage-like metrics
    if 'Margin' in col or 'ReturnOn' in col or 'TaxRate' in col or 'Ratio' in col and col not in ['currentRatio', 'quickRatio', 'cashRatio', 'debtEquityRatio', 'debtRatio', 'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowSalesRatio']:
        st.caption("Enter as decimal (e.g., 0.1 for 10%)")
    
    financial_inputs[col] = st.number_input(
        f"Enter value for {col}",
        min_value=min_values.get(col, 0.0), # Use .get() with a fallback for safety
        max_value=max_values.get(col, 1000.0),
        value=default_values.get(col, 0.0),
        step=step_values.get(col, 0.01),
        key=f"fin_{col}"
    )

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
        # The sentiment features (Avg_Positive, Avg_Neutral, Avg_Negative, Avg_Compound)
        # need to be derived from the TextBlob polarity and subjectivity
        # Ensure this logic matches how these features were created in your training data
        
        # Simple mapping for demonstration. Adjust if your training data's sentiment features
        # were derived differently (e.g., from a more complex NLP model).
        avg_positive = 1 if sentiment_result['polarity'] > 0.1 else 0
        avg_neutral = 1 if -0.1 <= sentiment_result['polarity'] <= 0.1 else 0
        avg_negative = 1 if sentiment_result['polarity'] < -0.1 else 0
        avg_compound = sentiment_result['polarity'] # Using TextBlob polarity as compound

        sentiment_data = {
            'Avg_Positive': [avg_positive],
            'Avg_Neutral': [avg_neutral],
            'Avg_Negative': [avg_negative],
            'Avg_Compound': [avg_compound]
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



