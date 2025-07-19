import streamlit as st
import pandas as pd
import joblib # For loading the scalers and models
from catboost import CatBoostClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Import VADER
import nltk # Import nltk

# --- NLTK Data Download (Crucial for VADER) ---
@st.cache_resource # Cache the download to run only once
def download_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError: # Catch LookupError for missing data
        st.info("Downloading VADER lexicon for sentiment analysis. This will only happen once.")
        nltk.download('vader_lexicon')
        st.success("VADER lexicon downloaded!")

download_nltk_vader()

# --- Configuration ---
st.set_page_config(page_title="Credit Rating & Sentiment Predictor", page_icon="📈", layout="centered")

# --- Define Feature Columns (Consistent with 5 removed metrics) ---
# Removed: debtEquityRatio, ebitPerRevenue, returnOnCapitalEmployed, operatingProfitMargin, companyEquityMultiplier
financial_cols = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin',
    'returnOnAssets', 'returnOnEquity', 'assetTurnover',
    'fixedAssetTurnover', 'debtRatio', 'effectiveTaxRate',
    'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare',
    'enterpriseValueMultiple', 'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio', 'payablesTurnover'
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
        st.warning("Please ensure 'CatboostML.modelA.cbm' is in the same directory and trained with the correct features.")

    try:
        scaler_fin = joblib.load('scaler_fin.pkl')
        scalers['fin'] = scaler_fin
        st.success("Scaler for Model A loaded successfully!")
    except Exception as e:
        st.error(f"Error loading scaler_fin.pkl: {e}")
        st.warning("Please ensure 'scaler_fin.pkl' is in the same directory and trained with the correct features.")

    # Load Model B (Financial + Sentiment) and its scaler
    try:
        model_B = CatBoostClassifier()
        model_B.load_model('CatboostML.modelB.cbm')
        models['B'] = model_B
        st.success("Model B (Financial + Sentiment) loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Model B: {e}")
        st.warning("Please ensure 'CatboostML.modelB.cbm' is in the same directory and trained with the correct features.")

    try:
        scaler_all = joblib.load('scaler_all.pkl')
        scalers['all'] = scaler_all
        st.success("Scaler for Model B loaded successfully!")
    except Exception as e:
        st.error(f"Error loading scaler_all.pkl: {e}")
        st.warning("Please ensure 'scaler_all.pkl' is in the same directory and trained with the correct features.")

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
    Analyzes the sentiment of a news article using VADER and returns the
    Avg_Positive, Avg_Neutral, Avg_Negative, and Avg_Compound scores
    as expected by your model.
    """
    if not news_article:
        return {'polarity': 0.0, 'subjectivity': 0.0, 'category': 'Neutral',
                'Avg_Positive': 0.0, 'Avg_Neutral': 1.0, 'Avg_Negative': 0.0, 'Avg_Compound': 0.0}

    try:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(news_article)
        
        compound_score = vs['compound']

        # Determine sentiment category based on compound score
        if compound_score >= 0.05:
            category = "Positive"
        elif compound_score <= -0.05:
            category = "Negative"
        else:
            category = "Neutral"
            
        # Generate Avg_Positive, Avg_Neutral, Avg_Negative as binary flags
        avg_positive_flag = 1.0 if compound_score >= 0.05 else 0.0
        avg_neutral_flag = 1.0 if -0.05 < compound_score < 0.05 else 0.0
        avg_negative_flag = 1.0 if compound_score <= -0.05 else 0.0

        return {'polarity': compound_score,
                'subjectivity': 0.0,
                'category': category,
                'Avg_Positive': avg_positive_flag,
                'Avg_Neutral': avg_neutral_flag,
                'Avg_Negative': avg_negative_flag,
                'Avg_Compound': compound_score}
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.0, 'category': 'Error',
                'Avg_Positive': 0.0, 'Avg_Neutral': 0.0, 'Avg_Negative': 0.0, 'Avg_Compound': 0.0}

# --- Streamlit UI ---

st.title("Company Financial Health & News Sentiment Analyzer")

# --- About Credit Ratings Section (using expander for a cleaner look) ---
with st.expander("About Credit Ratings & This Website"):
    st.markdown("""
    ### What is a Credit Rating?
    A credit rating is an independent assessment of a company's financial strength and its ability to meet its financial obligations. These ratings are crucial for investors, lenders, and businesses as they provide a quick snapshot of creditworthiness, influencing borrowing costs and investment decisions. Ratings typically range from 'AAA' (highest quality, lowest risk) to 'D' (default).

    ### How are Credit Ratings Calculated?
    Credit rating agencies use a comprehensive approach, combining quantitative financial analysis with qualitative factors.
    * **Financial Metrics (Quantitative):** This includes analyzing a company's balance sheet, income statement, and cash flow statement. Key ratios like liquidity (e.g., current ratio), profitability (e.g., gross profit margin), leverage (e.g., debt ratio), and efficiency (e.g., days of sales outstanding) are vital.
    * **News Sentiment (Qualitative):** Public sentiment and news coverage can significantly impact a company's perceived risk. Positive news might signal stability and growth, while negative news could indicate potential challenges.

    This application uses two machine learning models:
    * **Model A (Financial Only):** Predicts credit ratings based solely on a set of key financial metrics.
    * **Model B (Financial + Sentiment):** Combines these financial metrics with sentiment analysis derived from news articles to provide a more holistic prediction.

    ### How to Use This Website
    1.  **Enter Company Name:** Provide the name of the company you are analyzing.
    2.  **Input Financial Metrics:** Fill in the values for the various financial ratios in the designated section. Ensure you enter percentage-like metrics (e.g., margins, returns) as decimals (e.g., 0.10 for 10%).
    3.  **Predict with Model A:** Click the "Predict with Model A" button to get a credit rating prediction based only on the financial data.
    4.  **Enter News Article (for Model B):** Provide a relevant news article about the company in the text area for Model B.
    5.  **Analyze Sentiment & Predict with Model B:** Click this button to first analyze the sentiment of the news article and then get a credit rating prediction that incorporates both financial and sentiment data.
    """)

st.markdown("---")

# --- Input Fields for Financial Metrics (Common to both models) ---
st.header("Enter Financial Metrics")

company_name = st.text_input("Company Name", "Example Corp")

# Create a dictionary to hold all financial inputs
financial_inputs = {}

# Define sensible default values and ranges for each financial metric
# These are crucial for user experience and preventing invalid inputs
# Values are based on general financial health indicators.
# Updated default_values, min_values, max_values, step_values to match new financial_cols
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


for col in financial_cols:
    st.write(f"**{col}**")
    # Add a note for percentage-like metrics
    if 'Margin' in col or 'ReturnOn' in col or 'TaxRate' in col or ('Ratio' in col and col not in ['currentRatio', 'quickRatio', 'cashRatio', 'debtEquityRatio', 'debtRatio', 'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowSalesRatio']):
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
                            "Example: The company announced record loss this quarter, exceeding all expectations and leading to a significant stock price decrease. However, concerns about market competition are rising.",
                            height=200)

if st.button("Analyze Sentiment & Predict with Model B"):
    sentiment_result = analyze_sentiment(news_article)
    
    st.subheader("News Article Sentiment:")
    # Display VADER's specific scores for clarity
    st.info(f"VADER Compound Score: {sentiment_result['Avg_Compound']:.2f} (Positive: {sentiment_result['Avg_Positive']:.2f}, Neutral: {sentiment_result['Avg_Neutral']:.2f}, Negative: {sentiment_result['Avg_Negative']:.2f})")

    if sentiment_result['category'] == "Positive":
        st.success(f"Sentiment Category: **{sentiment_result['category']}** 😊")
    elif sentiment_result['category'] == "Negative":
        st.error(f"Sentiment Category: **{sentiment_result['category']}** 😠")
    else:
        st.info(f"Sentiment Category: **{sentiment_result['category']}** 😐")

    # Prepare data for Model B prediction
    if 'B' in models and 'all' in scalers:
        # Use the derived VADER scores directly as your sentiment features
        avg_positive = sentiment_result['Avg_Positive']
        avg_neutral = sentiment_result['Avg_Neutral']
        avg_negative = sentiment_result['Avg_Negative']
        avg_compound = sentiment_result['Avg_Compound']

        # Combine all inputs into a single dictionary
        all_inputs = {**financial_inputs,
                      'Avg_Positive': avg_positive,
                      'Avg_Neutral': avg_neutral,
                      'Avg_Negative': avg_negative,
                      'Avg_Compound': avg_compound}
        
        # Create a DataFrame from the combined dictionary, ensuring correct order
        combined_df_row = pd.DataFrame([all_inputs])[all_cols] # Explicitly select and order columns
        
        with st.spinner("Predicting with Model B..."):
            predicted_rating_B = predict_credit_rating(models['B'], scalers['all'], combined_df_row, all_cols)
            st.subheader(f"Predicted Credit Rating (Model B) for {company_name}:")
            st.success(f"**{predicted_rating_B}**")
    else:
        st.warning("Model B or its scaler not loaded. Cannot perform prediction.")

st.markdown("---")
st.info("Developed with Streamlit by your AI assistant. Remember to train and save your models and scalers!")



