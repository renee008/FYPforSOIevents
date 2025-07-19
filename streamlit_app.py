import streamlit as st
import pandas as pd
import joblib # For loading the scalers and models
from catboost import CatBoostClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Import VADER
import nltk # Import nltk
import shap # For explainability
import numpy as np # Import numpy for array handling
import matplotlib.pyplot as plt # For plotting feature contributions

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

# --- Define Feature Columns (Consistent with 5 removed metrics) ---
# Removed: debtEquityRatio, ebitPerRevenue, returnOnCapitalEmployed, operatingProfitMargin, companyEquityMultiplier
financial_cols = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin',
    'returnOnAssets', 'returnOnEquity', 'assetTurnover',
    'fixedAssetTurnover', 'debtRatio', 'effectiveTaxRate',
    'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare',
    'enterpriseValueMultiple', 'payablesTurnover','operatingCashFlowPerShare', 'operatingCashFlowSalesRatio'
]

sentiment_cols = ['Avg_Positive', 'Avg_Neutral', 'Avg_Negative', 'Avg_Compound']
all_cols = financial_cols + sentiment_cols

# --- Model and Scaler Loading ---
@st.cache_resource # Cache the models and scalers to avoid reloading on every rerun
def load_models_and_scalers():
    models = {}
    scalers = {}

    # Load Model A (Financial Only) and its scaler
    try:
        model_A = CatBoostClassifier()
        model_A.load_model('CatboostML.modelA.cbm')
        models['A'] = model_A
    except Exception as e:
        st.error(f"Error loading Model A: {e}")
        st.warning("Please ensure 'CatboostML.modelA.cbm' is in the same directory and trained with the correct features.")

    try:
        scaler_fin = joblib.load('scaler_fin.pkl')
        scalers['fin'] = scaler_fin
    except Exception as e:
        st.error(f"Error loading scaler_fin.pkl: {e}")
        st.warning("Please ensure 'scaler_fin.pkl' is in the same directory and trained with the correct features.")

    # Load Model B (Financial + Sentiment) and its scaler
    try:
        model_B = CatBoostClassifier()
        model_B.load_model('CatboostML.modelB.cbm')
        models['B'] = model_B
    except Exception as e:
        st.error(f"Error loading Model B: {e}")
        st.warning("Please ensure 'CatboostML.modelB.cbm' is in the same directory and trained with the correct features.")

    try:
        scaler_all = joblib.load('scaler_all.pkl')
        scalers['all'] = scaler_all
    except Exception as e:
        st.error(f"Error loading scaler_all.pkl: {e}")
        st.warning("Please ensure 'scaler_all.pkl' is in the same directory and trained with the correct features.")

    return models, scalers

models, scalers = load_models_and_scalers()

# --- Prediction Functions ---

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
        
        # Use .item() to extract the scalar string value from the numpy array
        predicted_rating = model.predict(scaled_data).item()
        probabilities = model.predict_proba(scaled_data)[0] # Get probabilities for the single instance
        
        # Map probabilities to class names
        class_names = model.classes_
        probabilities_dict = dict(zip(class_names, probabilities))
        
        return str(predicted_rating), probabilities_dict

    except Exception as e:
        st.error(f"Error during credit rating prediction: {e}")
        return "Prediction failed.", {}

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
def get_feature_contributions(model, scaler, input_df, feature_columns, top_n=5):
    """
    Calculates SHAP values and returns the top N most influential features
    and their impact (positive/negative) for the predicted class.
    """
    try:
        # Ensure feature_columns is a list
        feature_columns_list = list(feature_columns)

        input_df_reindexed = input_df[feature_columns_list]
        scaled_data = scaler.transform(input_df_reindexed)
        
        # Convert scaled_data back to a DataFrame with column names for SHAP
        scaled_data_df = pd.DataFrame(scaled_data, columns=feature_columns_list)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data_df) # Pass the DataFrame here

        # Get the predicted class label (e.g., 'B', 'AAA')
        predicted_class_label = model.predict(scaled_data).item()
        
        shap_values_for_prediction = None

        # Determine the integer index of the predicted class
        try:
            predicted_class_index = list(model.classes_).index(predicted_class_label)
        except ValueError:
            st.warning(f"Predicted class '{predicted_class_label}' not found in model.classes_ for SHAP explanation.")
            return []

        # Handle different SHAP output structures for multi-class CatBoost
        if isinstance(shap_values, list) and len(shap_values) == len(model.classes_):
            # This is the standard multi-class output: list of arrays, one for each class
            # Each array is (num_instances, num_features). We need the first instance.
            if predicted_class_index < len(shap_values):
                shap_values_for_prediction = shap_values[predicted_class_index][0] 
            else:
                st.warning(f"Predicted class index {predicted_class_index} out of bounds for SHAP values list (len: {len(shap_values)}).")
                return []
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # This handles the (1, N_FEATURES, N_CLASSES) structure
            # Extract SHAP values for the single instance (0) and the specific predicted class
            shap_values_for_prediction = shap_values[0, :, predicted_class_index]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2 and shap_values.shape[0] == 1:
            # Binary classification or simplified multi-class (1, N_FEATURES)
            shap_values_for_prediction = shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 1:
            # Already 1D array (N_FEATURES)
            shap_values_for_prediction = shap_values
        else:
            st.warning(f"Unexpected SHAP numpy array structure. ndim: {getattr(shap_values, 'ndim', 'N/A')}, shape: {getattr(shap_values, 'shape', 'N/A')}. Expected list of arrays or specific numpy array shapes.")
            return []

        if shap_values_for_prediction is None:
            return []
        
        # Final check: ensure the length of SHAP values matches the number of features
        if len(shap_values_for_prediction) != len(feature_columns_list):
            st.warning(f"Mismatch in SHAP values length ({len(shap_values_for_prediction)}) and feature columns length ({len(feature_columns_list)}). This can cause incorrect explanations.")
            return []

        # Create a Series for easier handling
        feature_impact = pd.Series(shap_values_for_prediction, index=feature_columns_list)
        
        # Sort by absolute value to find the most impactful features
        sorted_impact = feature_impact.abs().sort_values(ascending=False)
        
        contributions = []
        for feature_name in sorted_impact.index[:top_n]:
            impact_value = feature_impact[feature_name]
            direction = "higher" if impact_value > 0 else "lower" # Changed to higher/lower for clarity
            contributions.append((feature_name, impact_value, direction))
            
        return contributions

    except Exception as e:
        st.error(f"Error calculating feature contributions: {e}")
        return []

# --- Plotting Function for SHAP Contributions ---
def plot_shap_contributions_chart(contributions, title):
    """
    Plots a horizontal bar chart of SHAP feature contributions.
    """
    if not contributions:
        return None # No plot to generate

    features = [c[0] for c in contributions]
    impacts = [c[1] for c in contributions]
    colors = ['#2ca02c' if c[2] == 'higher' else '#d62728' for c in contributions] # Green for higher, Red for lower

    # Sort by absolute impact for consistent display (already sorted by get_feature_contributions, but re-confirm for plot)
    # This re-sorts just in case, ensuring plot order matches text order
    sorted_indices = np.argsort(np.abs(impacts))[::-1]
    features_sorted = [features[i] for i in sorted_indices]
    impacts_sorted = [impacts[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]

    # Adjust figure size dynamically based on number of features
    fig, ax = plt.subplots(figsize=(10, max(4, len(features_sorted) * 0.6 + 1))) 
    y_pos = np.arange(len(features_sorted))

    ax.barh(y_pos, impacts_sorted, color=colors_sorted)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted)
    ax.invert_yaxis() # Features with highest impact at the top
    ax.set_xlabel("SHAP Value (Impact on Prediction)")
    ax.set_title(title)
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--') # Add a vertical line at 0 for reference
    plt.tight_layout()
    return fig


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
    st.session_state.financial_inputs = {col: default_values.get(col, 0.0) for col in financial_cols}
if 'news_article' not in st.session_state:
    st.session_state.news_article = "Example: The company announced record loss this quarter, exceeding all expectations and leading to a significant stock price decrease. However, concerns about market competition are rising."
if 'company_name' not in st.session_state:
    st.session_state.company_name = "Example Corp"
if 'predict_A_triggered' not in st.session_state:
    st.session_state.predict_A_triggered = False
if 'predict_B_triggered' not in st.session_state:
    st.session_state.predict_B_triggered = False

def reset_inputs():
    st.session_state.financial_inputs = {col: default_values.get(col, 0.0) for col in financial_cols}
    st.session_state.news_article = "Example: The company announced record loss this quarter, exceeding all expectations and leading to a significant stock price decrease. However, concerns about market competition are rising."
    st.session_state.company_name = "Example Corp"
    st.session_state.predict_A_triggered = False
    st.session_state.predict_B_triggered = False

def run_all_predictions():
    st.session_state.predict_A_triggered = True
    st.session_state.predict_B_triggered = True


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

    This application uses two machine learning models:
    * **Model A (Financial Only):** Predicts credit ratings based solely on a set of key financial metrics.
    * **Model B (Financial + Sentiment):** Combines these financial metrics with sentiment analysis derived from news articles to provide a more holistic prediction.
    """)

with tab_how_to_use:
    st.markdown("""
    ### How to Use This Website
    1.  **Enter Company Name:** Provide the name of the company you are analyzing in the text input field below.
    2.  **Input Financial Metrics:** Fill in the values for the various financial ratios in the designated section. Please ensure you enter percentage-like metrics (e.g., margins, returns) as decimals (e.g., enter `0.10` for 10%).
        * **Note on Numerical Inputs:** The number input fields require manual typing of values. The increment/decrement arrows are hidden for a cleaner interface.
    3.  **Predict with Models:** You can either click "Predict with Model A" or "Analyze Sentiment & Predict with Model B" for individual predictions, or click "Predict All Models" to run both at once.
    4.  **Review Results:** The prediction results, probabilities, and key feature contributions will appear in the "Prediction Results" section.
    5.  **Reset Inputs:** Use the "Reset All Inputs" button to clear the form and start fresh.
    """)

st.markdown("---") # Separator below the tabs

st.header("Enter Company Details")

st.session_state.company_name = st.text_input("Company Name", value=st.session_state.company_name, key="company_name_input")

st.markdown("---")

# --- Input Fields for Financial Metrics ---
st.header("Enter Financial Metrics")

# Display inputs in a single column
for i, col_name in enumerate(financial_cols):
    st.write(f"**{col_name}**")
    # Add a note for percentage-like metrics
    if 'Margin' in col_name or 'ReturnOn' in col_name or 'TaxRate' in col_name or ('Ratio' in col_name and col_name not in ['currentRatio', 'quickRatio', 'cashRatio', 'debtRatio', 'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowSalesRatio']):
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
    st.info(metric_help_text.get(col_name, "No specific help text available."), icon="�")

    st.session_state.financial_inputs[col_name] = st.number_input(
        f"Value for {col_name}",
        min_value=min_values.get(col_name, 0.0),
        max_value=max_values.get(col_name, 1000.0),
        value=st.session_state.financial_inputs.get(col_name, default_values.get(col_name, 0.0)),
        step=step_values.get(col_name, 0.01), # Still need step for internal logic/validation even if arrows are hidden
        key=f"fin_input_{col_name}" # Unique key for each input
    )

# Convert financial inputs to a DataFrame row
financial_df_row = pd.DataFrame([st.session_state.financial_inputs])

st.markdown("---")

# --- Prediction Buttons ---
col_predict_A, col_predict_B, col_predict_all = st.columns(3)

with col_predict_A:
    if st.button("Predict with Model A", key="trigger_predict_A"):
        st.session_state.predict_A_triggered = True
        st.session_state.predict_B_triggered = False # Reset other trigger
with col_predict_B:
    if st.button("Analyze Sentiment & Predict with Model B", key="trigger_predict_B"):
        st.session_state.predict_B_triggered = True
        st.session_state.predict_A_triggered = False # Reset other trigger
with col_predict_all:
    if st.button("Predict All Models", key="trigger_predict_all", on_click=run_all_predictions):
        pass # The on_click function handles state updates

st.button("Reset All Inputs", on_click=reset_inputs)

st.markdown("---")

# --- Prediction Results Section ---
st.header("Prediction Results")

# Model A Prediction Display
if st.session_state.predict_A_triggered:
    st.subheader("1. Model A (Financial Only) Prediction")
    if 'A' in models and 'fin' in scalers:
        with st.spinner("Predicting with Model A..."):
            predicted_rating_A, probabilities_A = predict_credit_rating(models['A'], scalers['fin'], financial_df_row, financial_cols)
            
            st.write(f"**Predicted Credit Rating for {st.session_state.company_name}:**")
            col_rating_A_display, col_popover_A_display = st.columns([0.7, 0.3])
            with col_rating_A_display:
                st.success(f"**{predicted_rating_A}**")
            with col_popover_A_display:
                with st.popover(f"What is '{predicted_rating_A}'?"):
                    st.write(f"**{predicted_rating_A}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating_A, 'Definition not available for this rating.')}")

            st.write("---")
            st.subheader("Prediction Probabilities:")
            prob_df_A = pd.DataFrame(probabilities_A.items(), columns=['Rating', 'Probability'])
            prob_df_A['Probability'] = prob_df_A['Probability'].astype(float)
            prob_df_A = prob_df_A.sort_values(by='Probability', ascending=False)
            prob_df_A['Probability'] = prob_df_A['Probability'].apply(lambda x: f"{x:.2%}")
            st.dataframe(prob_df_A, hide_index=True, use_container_width=True)

            st.write("---")
            st.subheader("Key Feature Contributions (Model A):")
            st.markdown("*(Green bars push the rating higher, red bars push it lower)*")
            contributions_A = get_feature_contributions(models['A'], scalers['fin'], financial_df_row, financial_cols)
            if contributions_A:
                for feature, value, direction in contributions_A:
                    if direction == "higher":
                        st.markdown(f"- **{feature}**: Pushed rating **higher** (Impact: {value:.4f})")
                    else:
                        st.markdown(f"- **{feature}**: Pushed rating **lower** (Impact: {value:.4f})")
                
                fig_A = plot_shap_contributions_chart(contributions_A, f"Top Feature Contributions for {st.session_state.company_name} (Model A)")
                if fig_A:
                    st.pyplot(fig_A)
                    plt.close(fig_A)
                else:
                    st.info("No sufficient data to plot feature contributions for Model A.")
            else:
                st.info("Could not determine feature contributions for Model A.")
    else:
        st.warning("Model A or its scaler not loaded. Cannot perform prediction.")
    st.markdown("---")


# Model B Prediction Display
if st.session_state.predict_B_triggered:
    st.subheader("2. Model B (Financial + Sentiment) Prediction")
    news_article_for_B = st.session_state.news_article # Use the current news article from session state
    sentiment_result_B = analyze_sentiment(news_article_for_B)
    
    st.write("News Article Sentiment:")
    st.info(f"VADER Compound Score: {sentiment_result_B['Avg_Compound']:.2f} (Positive: {sentiment_result_B['Avg_Positive']:.2f}, Neutral: {sentiment_result_B['Avg_Neutral']:.2f}, Negative: {sentiment_result_B['Avg_Negative']:.2f})")

    if sentiment_result_B['category'] == "Positive":
        st.success(f"Sentiment Category: **{sentiment_result_B['category']}** 😊")
    elif sentiment_result_B['category'] == "Negative":
        st.error(f"Sentiment Category: **{sentiment_result_B['category']}** 😠")
    else:
        st.info(f"Sentiment Category: **{sentiment_result_B['category']}** 😐")

    if 'B' in models and 'all' in scalers:
        avg_positive = sentiment_result_B['Avg_Positive']
        avg_neutral = sentiment_result_B['Avg_Neutral']
        avg_negative = sentiment_result_B['Avg_Negative']
        avg_compound = sentiment_result_B['Avg_Compound']

        all_inputs = {**st.session_state.financial_inputs,
                      'Avg_Positive': avg_positive,
                      'Avg_Neutral': avg_neutral,
                      'Avg_Negative': avg_negative,
                      'Avg_Compound': avg_compound}
        
        combined_df_row = pd.DataFrame([all_inputs])[all_cols]
        
        with st.spinner("Predicting with Model B..."):
            predicted_rating_B, probabilities_B = predict_credit_rating(models['B'], scalers['all'], combined_df_row, all_cols)
            
            st.write(f"**Predicted Credit Rating for {st.session_state.company_name}:**")
            col_rating_B_display, col_popover_B_display = st.columns([0.7, 0.3])
            with col_rating_B_display:
                st.success(f"**{predicted_rating_B}**")
            with col_popover_B_display:
                with st.popover(f"What is '{predicted_rating_B}'?"):
                    st.write(f"**{predicted_rating_B}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating_B, 'Definition not available for this rating.')}")

            st.write("---")
            st.subheader("Prediction Probabilities:")
            prob_df_B = pd.DataFrame(probabilities_B.items(), columns=['Rating', 'Probability'])
            prob_df_B['Probability'] = prob_df_B['Probability'].astype(float)
            prob_df_B = prob_df_B.sort_values(by='Probability', ascending=False)
            prob_df_B['Probability'] = prob_df_B['Probability'].apply(lambda x: f"{x:.2%}")
            st.dataframe(prob_df_B, hide_index=True, use_container_width=True)

            st.write("---")
            st.subheader("Key Feature Contributions (Model B):")
            st.markdown("*(Green bars push the rating higher, red bars push it lower)*")
            contributions_B = get_feature_contributions(models['B'], scalers['all'], combined_df_row, all_cols)
            if contributions_B:
                for feature, value, direction in contributions_B:
                    if direction == "higher":
                        st.markdown(f"- **{feature}**: Pushed rating **higher** (Impact: {value:.4f})")
                    else:
                        st.markdown(f"- **{feature}**: Pushed rating **lower** (Impact: {value:.4f})")
                
                fig_B = plot_shap_contributions_chart(contributions_B, f"Top Feature Contributions for {st.session_state.company_name} (Model B)")
                if fig_B:
                    st.pyplot(fig_B)
                    plt.close(fig_B)
                else:
                    st.info("No sufficient data to plot feature contributions for Model B.")
            else:
                st.info("Could not determine feature contributions for Model B.")
    else:
        st.warning("Model B or its scaler not loaded. Cannot perform prediction.")

st.markdown("---")
st.info("Developed with Streamlit by your AI assistant.")
