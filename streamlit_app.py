import streamlit as st
import pandas as pd
import joblib # For loading the scalers and models
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Import VADER
import nltk # Import nltk
import shap # For explainability
import numpy as np # Import numpy for array handling
import matplotlib.pyplot as plt # For plotting SHAP contributions

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
        box_shadow: 4px 4px 10px rgba(0,0,0,0.3);
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
st.set_page_config(page_title="Credit Rating & Sentiment Predictor", page_icon="📈", layout="centered")

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

SENTIMENT_COLS = ['Avg_Positive', 'Avg_Neutral', 'Avg_Negative', 'Avg_Compound']
ALL_COLS = FINANCIAL_COLS + SENTIMENT_COLS

# --- Define a consistent order for credit ratings ---
RATING_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']

# --- Model and Scaler Loading ---
@st.cache_resource # Cache the models and scalers to avoid reloading on every rerun
def load_models_and_scalers():
    models = {}
    scalers = {}

    try:
        # Load CatBoost Model A (Financial Only) and its scaler (renee_scaler_fin.pkl)
        cat_model_A = CatBoostClassifier()
        cat_model_A.load_model('CatboostML.modelA.cbm')
        models['CatBoost Model A (Financial Only)'] = cat_model_A
        scalers['CatBoost Model A (Financial Only)'] = joblib.load('renee_scaler_financial.pkl') 

        # Load CatBoost Model B (Financial + Sentiment) and its scaler (renee_scaler_all.pkl)
        cat_model_B = CatBoostClassifier()
        cat_model_B.load_model('CatboostML.modelB.cbm')
        models['CatBoost Model B (Financial + Sentiment)'] = cat_model_B
        scalers['CatBoost Model B (Financial + Sentiment)'] = joblib.load('renee_scaler_all.pkl') 

        # Load RandomForest Model A (Financial Only) and its scaler (ath_scaler_fin.pkl)
        rf_model_A = joblib.load('RandomForest_modelA.pkl')
        models['RandomForest Model A (Financial Only)'] = rf_model_A
        scalers['RandomForest Model A (Financial Only)'] = joblib.load('ath_scaler_fin.pkl') 

        # Load RandomForest Model B (Financial + Sentiment) and its scaler (ath_scaler_all.pkl)
        rf_model_B = joblib.load('RandomForest_modelB.pkl')
        models['RandomForest Model B (Financial + Sentiment)'] = rf_model_B
        scalers['RandomForest Model B (Financial + Sentiment)'] = joblib.load('ath_scaler_all.pkl') 

        st.success("All models and scalers loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler file: {e}. Please ensure all model and scaler files are in the same directory.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()
    return models, scalers

# Load models and scalers at the start of the app
models, scalers = load_models_and_scalers()

# --- Prediction Function ---
def predict_credit_rating(model, scaler, input_df, feature_columns) -> tuple:
    """
    Predicts the credit rating and its probabilities using the given model and scaler.
    Returns (predicted_rating, probabilities_dict).
    """
    if model is None or scaler is None:
        return "Model or scaler not loaded. Cannot predict credit rating.", {}

    try:
        input_df_reindexed = input_df[feature_columns]
        scaled_data = scaler.transform(input_df_reindexed)
        
        # Predict the rating
        # CatBoost's predict returns a 2D array, RandomForest's predict returns a 1D array
        predicted_rating = model.predict(scaled_data)
        if isinstance(predicted_rating, np.ndarray) and predicted_rating.ndim > 0:
            predicted_rating = predicted_rating.flatten()[0] # Get the scalar value

        # Get prediction probabilities
        probabilities = model.predict_proba(scaled_data)[0] # Get probabilities for the single instance
        
        # Map probabilities to class names
        # Ensure model.classes_ is consistent with RATING_ORDER if possible
        class_names = model.classes_ if hasattr(model, 'classes_') else RATING_ORDER
        probabilities_dict = dict(zip(class_names, probabilities))
        
        return str(predicted_rating), probabilities_dict

    except Exception as e:
        st.error(f"Error during credit rating prediction: {e}")
        return "Prediction failed.", {}

# --- Sentiment Analysis Function ---
def analyze_sentiment(news_article: str) -> dict:
    """
    Analyzes the sentiment of a news article using VADER and returns the
    Avg_Positive, Avg_Neutral, Avg_Negative, and Avg_Compound scores.
    """
    if not news_article:
        return {'polarity': 0.0, 'subjectivity': 0.0, 'category': 'Neutral',
                'Avg_Positive': 0.0, 'Avg_Neutral': 1.0, 'Avg_Negative': 0.0, 'Avg_Compound': 0.0}

    try:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(news_article)
        
        compound_score = vs['compound']

        if compound_score >= 0.05:
            category = "Positive"
        elif compound_score <= -0.05:
            category = "Negative"
        else:
            category = "Neutral"
            
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

# --- SHAP Explainability Function ---
def plot_shap_contributions(model, scaler, input_df, feature_columns, predicted_rating_str):
    """
    Calculates and plots SHAP values for a single prediction.
    """
    try:
        input_df_reindexed = input_df[feature_columns]
        scaled_data = scaler.transform(input_df_reindexed)

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data)

        # Determine which SHAP values to use for plotting based on predicted class
        if isinstance(shap_values, list): # Multi-class output (e.g., CatBoost or some RandomForest setups)
            # Find the index of the predicted class in the model's classes_ array
            if hasattr(model, 'classes_'):
                try:
                    predicted_class_idx = list(model.classes_).index(predicted_rating_str)
                except ValueError:
                    st.warning(f"Predicted class '{predicted_rating_str}' not found in model.classes_ for SHAP explanation. Using average SHAP values.")
                    predicted_class_idx = 0 # Fallback to first class index
            else: # Fallback if model.classes_ is not available (e.g., some custom RF setups)
                st.warning("Model classes not found for precise SHAP explanation. Using first class SHAP values.")
                predicted_class_idx = 0 # Default to first class's explanation
            
            shap_values_for_plot = shap_values[predicted_class_idx][0] # Get SHAP values for the single instance of the predicted class
            
        else: # Binary classification or simplified multi-class (e.g., RandomForest might return 2D array directly)
            shap_values_for_plot = shap_values[0] # Get SHAP values for the single instance

        # Create a SHAP Explanation object for plotting
        shap_explanation = shap.Explanation(
            values=shap_values_for_plot,
            base_values=explainer.expected_value[predicted_class_idx] if isinstance(explainer.expected_value, np.ndarray) and isinstance(shap_values, list) else explainer.expected_value,
            data=input_df_reindexed.iloc[0].values, # Use the original input data for plotting
            feature_names=feature_columns
        )

        # Generate the SHAP summary plot (bar plot for single instance)
        fig, ax = plt.subplots(figsize=(10, len(feature_columns) * 0.4 + 2)) # Dynamic height
        shap.summary_plot(shap_values_for_plot, input_df_reindexed, feature_names=feature_columns, plot_type="bar", show=False, color='#2ca02c') # Green color
        ax.set_title(f"Feature Contributions for Predicted Rating: {predicted_rating_str}")
        ax.set_xlabel("SHAP Value (Impact on Prediction)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        st.error(f"Could not generate SHAP plot: {e}. This might happen if the model or data structure is not fully compatible with SHAP's TreeExplainer or if there's an issue with the input data.")
        st.info("Ensure the model was trained with the same feature names and order, and that the SHAP library is compatible with your model version.")


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
if 'news_article' not in st.session_state:
    st.session_state.news_article = "Example: The company announced record loss this quarter, exceeding all expectations and leading to a significant stock price decrease. However, concerns about market competition are rising."
if 'company_name' not in st.session_state:
    st.session_state.company_name = "Example Corp"
if 'last_predicted_model' not in st.session_state:
    st.session_state.last_predicted_model = None
if 'last_predicted_rating' not in st.session_state:
    st.session_state.last_predicted_rating = None
if 'last_input_df' not in st.session_state:
    st.session_state.last_input_df = None
if 'last_feature_cols' not in st.session_state:
    st.session_state.last_feature_cols = None
if 'last_scaler' not in st.session_state:
    st.session_state.last_scaler = None


def reset_inputs():
    st.session_state.financial_inputs = {col: default_values.get(col, 0.0) for col in FINANCIAL_COLS}
    st.session_state.news_article = "Example: The company announced record loss this quarter, exceeding all expectations and leading to a significant stock price decrease. However, concerns about market competition are rising."
    st.session_state.company_name = "Example Corp"
    st.session_state.last_predicted_model = None
    st.session_state.last_predicted_rating = None
    st.session_state.last_input_df = None
    st.session_state.last_feature_cols = None
    st.session_state.last_scaler = None


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

    This application uses four machine learning models:
    * **CatBoost Model A (Financial Only):** Predicts credit ratings based solely on a set of key financial metrics using a CatBoost Classifier.
    * **CatBoost Model B (Financial + Sentiment):** Combines financial metrics with sentiment analysis from news articles using a CatBoost Classifier.
    * **RandomForest Model A (Financial Only):** Predicts credit ratings based solely on a set of key financial metrics using a RandomForest Classifier.
    * **RandomForest Model B (Financial + Sentiment):** Combines financial metrics with sentiment analysis from news articles using a RandomForest Classifier.
    """)

with tab_how_to_use:
    st.markdown("""
    ### How to Use This Website
    1.  **Enter Company Name:** Provide the name of the company you are analyzing.
    2.  **Select a Model:** Choose one of the four available models from the dropdown.
    3.  **Input Financial Metrics:** Fill in the values for the various financial ratios. Enter percentage-like metrics (e.g., margins, returns) as decimals (e.g., `0.10` for 10%).
    4.  **Enter News Article (for Model B types):** If you select a "Financial + Sentiment" model (Model B), provide a relevant news article.
    5.  **Predict:** Click the "Predict Credit Rating" button to get a prediction and feature contributions.
    6.  **Review Results:** The predicted rating, probabilities, and key feature contributions will appear.
    7.  **Reset Inputs:** Use the "Reset All Inputs" button to clear the form and start fresh.
    """)

st.markdown("---") # Separator below the tabs

st.header("Enter Company Details")

st.session_state.company_name = st.text_input("Company Name", value=st.session_state.company_name, key="company_name_input")

st.markdown("---")

# --- Model Selection ---
st.header("Select Prediction Model")
selected_model_name = st.selectbox(
    "Choose a model for prediction:",
    list(models.keys()),
    key="model_selector"
)
selected_model = models[selected_model_name]
selected_scaler = scalers[selected_model_name]

# Determine which features are needed for the selected model
if "Financial Only" in selected_model_name:
    required_features = FINANCIAL_COLS
    st.info(f"You selected '{selected_model_name}'. This model uses only financial metrics.")
    sentiment_input_needed = False
else: # Implies "Financial + Sentiment"
    required_features = ALL_COLS
    st.info(f"You selected '{selected_model_name}'. This model uses both financial metrics and news sentiment.")
    sentiment_input_needed = True

st.markdown("---")

# --- Input Fields for Financial Metrics ---
st.header("Enter Financial Metrics")
st.markdown("*(All values should be numerical. Enter percentages as decimals, e.g., 0.1 for 10%)*")

# Use columns for better layout of inputs
num_cols_per_row = 2
cols = st.columns(num_cols_per_row)

for i, col_name in enumerate(FINANCIAL_COLS): # Always show all financial inputs
    with cols[i % num_cols_per_row]:
        st.write(f"**{col_name}**")
        # Add a note for percentage-like metrics
        if 'Margin' in col_name or 'ReturnOn' in col_name or 'TaxRate' in col_name or 'Ratio' in col_name:
            st.caption("Enter as decimal (e.g., 0.1 for 10%)")
        
        # Add a tooltip/help text for each metric
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

# --- Sentiment Input (only if a Model B type is selected) ---
sentiment_result = None
if sentiment_input_needed:
    st.header("Enter News Article (for Sentiment Analysis)")
    st.session_state.news_article = st.text_area("Enter Company News Article Here",
                                value=st.session_state.news_article,
                                height=200, key="news_article_input")
    
    sentiment_result = analyze_sentiment(st.session_state.news_article)
    
    st.subheader("News Article Sentiment:")
    st.info(f"VADER Compound Score: {sentiment_result['Avg_Compound']:.2f} (Positive: {sentiment_result['Avg_Positive']:.2f}, Neutral: {sentiment_result['Avg_Neutral']:.2f}, Negative: {sentiment_result['Avg_Negative']:.2f})")

    if sentiment_result['category'] == "Positive":
        st.success(f"Sentiment Category: **{sentiment_result['category']}** 😊")
    elif sentiment_result['category'] == "Negative":
        st.error(f"Sentiment Category: **{sentiment_result['category']}** 😠")
    else:
        st.info(f"Sentiment Category: **{sentiment_result['category']}** 😐")

st.markdown("---")

# --- Prediction Button ---
if st.button(f"Predict Credit Rating with {selected_model_name}", key="predict_button"):
    
    # Prepare input data based on selected model type
    if sentiment_input_needed:
        if sentiment_result is None: # Should not happen if sentiment_input_needed is True
            st.error("Sentiment analysis could not be performed. Please ensure a news article is entered.")
            st.stop()

        all_inputs = {**st.session_state.financial_inputs,
                      'Avg_Positive': sentiment_result['Avg_Positive'],
                      'Avg_Neutral': sentiment_result['Avg_Neutral'],
                      'Avg_Negative': sentiment_result['Avg_Negative'],
                      'Avg_Compound': sentiment_result['Avg_Compound']}
        input_df_for_prediction = pd.DataFrame([all_inputs])
    else:
        input_df_for_prediction = financial_df_row.copy() # Only financial data

    # Ensure columns are in the correct order as expected by the scaler and model
    input_df_for_prediction = input_df_for_prediction[required_features]

    with st.spinner(f"Predicting with {selected_model_name}..."):
        predicted_rating, probabilities = predict_credit_rating(selected_model, selected_scaler, input_df_for_prediction, required_features)
        
        st.header("3. Prediction Result")
        st.success(f"The predicted credit rating for {st.session_state.company_name} using {selected_model_name} is: **{predicted_rating}**")

        st.subheader("Prediction Probabilities:")
        prob_df = pd.DataFrame(probabilities.items(), columns=['Rating', 'Probability'])
        prob_df['Probability'] = prob_df['Probability'].astype(float) # Ensure numeric for sorting
        prob_df = prob_df.sort_values(by='Probability', ascending=False)
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}") # Format as percentage
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

        st.write("---")
        st.subheader("Key Feature Contributions (SHAP Values):")
        st.markdown("*(Green bars indicate a positive contribution to the predicted rating; red bars indicate a negative contribution.)*")
        
        # Store last prediction details in session state for SHAP plotting
        st.session_state.last_predicted_model = selected_model
        st.session_state.last_predicted_rating = predicted_rating
        st.session_state.last_input_df = input_df_for_prediction
        st.session_state.last_feature_cols = required_features
        st.session_state.last_scaler = selected_scaler

        # Call the SHAP plotting function
        plot_shap_contributions(
            st.session_state.last_predicted_model,
            st.session_state.last_scaler,
            st.session_state.last_input_df,
            st.session_state.last_feature_cols,
            st.session_state.last_predicted_rating
        )

# --- Reset Button (placed at the bottom for accessibility) ---
st.markdown("---")
st.button("Reset All Inputs", on_click=reset_inputs)

st.markdown("---")
st.info("Developed with Streamlit by your AI assistant.")

