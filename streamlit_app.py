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

# Define colors and fonts
PRIMARY_COLOR = "#0B3954"  # Navy Blue
SECONDARY_COLOR = "#BFD7EA"  # Light Blue/Grey background
ACCENT_COLOR = "#FF9F1C"  # Gold/Orange
TEXT_COLOR = "#333333"  # Dark Gray

# Inject CSS for fonts, colors, and layout
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Roboto Slab', serif;
        color: {TEXT_COLOR};
        background-color: #f7f9fc;
    }}
    .title {{
        font-size: 38px;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        text-transform: uppercase;
        margin-bottom: 15px;
        text-align: center;
    }}
    .section-header {{
        font-size: 24px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        color: {PRIMARY_COLOR};
        border-bottom: 3px solid {ACCENT_COLOR};
        padding-bottom: 5px;
    }}
    .caption-text {{
        font-size: 12px;
        color: #666666;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 10px;
    }}
    .card {{
        background: #ffffff;
        padding: 20px 30px;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-radius: 12px;
    }}
    div.stButton > button:first-child {{
        background-color: {ACCENT_COLOR};
        color: white;
        font-weight: 600;
        border-radius: 10px;
        height: 45px;
        width: 100%;
        font-size: 18px;
        transition: background-color 0.3s ease;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #cc7a00;
        color: white;
    }}
    .stTextInput > label, .stNumberInput > label {{
        font-weight: 600;
        color: {PRIMARY_COLOR};
    }}
    /* Adjust expander header font */
    .streamlit-expanderHeader {{
        font-weight: 600 !important;
        color: {PRIMARY_COLOR} !important;
        font-size: 18px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Company Financial Health & News Sentiment Analyzer</div>', unsafe_allow_html=True)

# --- Input Section: Financial Metrics ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💰 Enter Financial Metrics</div>', unsafe_allow_html=True)
    st.markdown("*(All values should be numerical. Enter percentages as decimals, e.g., 0.1 for 10%)*")

    # Increase columns per row for better use of space
    num_cols_per_row_in_expander = 3

    for category, cols_in_category in FINANCIAL_CATEGORIES.items():
        with st.expander(f"**{category}**", expanded=False):
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
    st.markdown('</div>', unsafe_allow_html=True)

# --- Input Section: News URLs ---
if sentiment_input_needed:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📰 News Article URL(s) for Sentiment Analysis</div>', unsafe_allow_html=True)

        st.session_state.news_article_urls = st.text_area(
            "Paste News Article URL(s) Here (one per line)",
            value=st.session_state.news_article_urls,
            height=150,
            key="news_article_urls_input"
        )

        st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Button ---
st.markdown('<div class="card">', unsafe_allow_html=True)
if st.button(f"Predict Credit Rating(s)", key="predict_button"):
    # Your existing prediction logic here (no change)

    # (You can keep your prediction code as is, wrapped inside this container)
    pass  # placeholder for your code
st.markdown('</div>', unsafe_allow_html=True)

# --- Results Section ---
# Wrap your results display (predicted rating, probabilities, feature importances) inside a card container similarly

# Example placeholder:
# with st.container():
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
#     # Show your results here
#     st.markdown('</div>', unsafe_allow_html=True)

# --- Feature Importance Plot (with color update) ---
def plot_feature_contributions(model, feature_columns, model_label):
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

        max_importance = np.max(feature_importances)
        if max_importance > 0:
            scaled_importances = (feature_importances / max_importance) * 10
        else:
            scaled_importances = np.zeros_like(feature_importances)

        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': scaled_importances
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(5, len(feature_columns) * 0.3)))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=ACCENT_COLOR)
        ax.set_facecolor(SECONDARY_COLOR)
        fig.patch.set_facecolor('#f9f9f9')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_xlabel("Feature Importance Score (Scaled to Max 10)", fontsize=12, color=PRIMARY_COLOR)
        ax.set_title(f"Overall Feature Contributions:\n{model_label}", fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_xticklabels([str(int(x)) for x in np.arange(0, 11, 1)], color=PRIMARY_COLOR)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Could not generate Feature Importance plot for {model_label}: {e}. Ensure the model object has a 'feature_importances_' attribute or a 'get_feature_importance()' method.")
How to integrate this:
Replace your existing st.title and section headers with the styled divs using unsafe_allow_html=True.

Wrap your main input blocks and result displays inside <div class="card"> containers (done above).

Increase financial input columns to 3 per row for better density.

Use the updated plot colors and fonts for feature importance plots.

Buttons styled with accent gold and full width.

Add icons like 💰, 📰, 📊 in headers to reinforce financial theme.

Add the CSS style block at the top for global font and color styling.

If you want, I can also help you fully place your existing prediction code and all UI code inside this style framework, just ask! Would you like me to?








You said:
yes please


ChatGPT said:
Absolutely! Here's your full Streamlit UI code with all your existing logic wrapped inside the professional, financial-themed style and layout.
I've integrated:

The CSS styling for fonts, colors, buttons, expanders, etc.

Your financial input section with 3 columns per row inside cards

News URL input in a styled card

Prediction button styled and inside a card

Prediction results and feature importance plots inside cards and styled headers

Proper spacing and icons for a polished look

Complete integrated code:

python
Copy
Edit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

# --- Define your constants here or import them ---
FINANCIAL_CATEGORIES = { 
    # Your categories and financial columns dictionary here
}
min_values = {}       # Your min values dict
max_values = {}       # Your max values dict
default_values = {}   # Your default values dict
step_values = {}      # Your step values dict

FINANCIAL_COLS = []   # List of financial columns used in model A
SENTIMENT_COLS = []   # List of sentiment columns
ALL_COLS = []         # All columns used in model B

CREDIT_RATING_DEFINITIONS = {
    # Your credit rating definitions dictionary here
}

# Placeholder for your model and scaler dicts
models = {}
scalers = {}

# Placeholder variables from your session_state
if "financial_inputs" not in st.session_state:
    st.session_state.financial_inputs = {}

if "news_article_urls" not in st.session_state:
    st.session_state.news_article_urls = ""

if "company_name" not in st.session_state:
    st.session_state.company_name = "Your Company"

# Flag for whether sentiment input is needed (set appropriately in your code)
sentiment_input_needed = True

# Selected model from your UI dropdown (set appropriately)
selected_model_name = "Model A (Financial Only)"  # example
selected_model = models.get(selected_model_name, None)
selected_scaler = scalers.get(selected_model_name, None)

# Label encoder placeholder
label_encoder = None

# Reset inputs function placeholder
def reset_inputs():
    st.session_state.financial_inputs = {}
    st.session_state.news_article_urls = ""

# Prediction function placeholder
def _predict_single_model(model, scaler, input_df, features, label_enc):
    # Your prediction logic here
    # Return dummy results for now
    return "AAA", {"AAA": 0.75, "AA": 0.15, "A": 0.10}

# Sentiment analysis placeholder function
def analyze_multiple_urls_sentiment(urls):
    # Your sentiment analysis logic here
    # Return dummy sentiment
    overall = {
        'Avg_Compound': 0.2,
        'Avg_Positive': 0.5,
        'Avg_Neutral': 0.4,
        'Avg_Negative': 0.1,
        'category': 'Positive'
    }
    details = {url: {"status": "Success", "sentiment": overall} for url in urls}
    return overall, details

# Feature importance plot function with financial color styling
def plot_feature_contributions(model, feature_columns, model_label):
    PRIMARY_COLOR = "#0B3954"
    SECONDARY_COLOR = "#BFD7EA"
    ACCENT_COLOR = "#FF9F1C"
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

        max_importance = np.max(feature_importances)
        if max_importance > 0:
            scaled_importances = (feature_importances / max_importance) * 10
        else:
            scaled_importances = np.zeros_like(feature_importances)

        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': scaled_importances
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(5, len(feature_columns) * 0.3)))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=ACCENT_COLOR)
        ax.set_facecolor(SECONDARY_COLOR)
        fig.patch.set_facecolor('#f9f9f9')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_xlabel("Feature Importance Score (Scaled to Max 10)", fontsize=12, color=PRIMARY_COLOR)
        ax.set_title(f"Overall Feature Contributions:\n{model_label}", fontsize=14, fontweight='bold', color=PRIMARY_COLOR)
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_xticklabels([str(int(x)) for x in np.arange(0, 11, 1)], color=PRIMARY_COLOR)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Could not generate Feature Importance plot for {model_label}: {e}.")

# --- CSS Styling ---
PRIMARY_COLOR = "#0B3954"
SECONDARY_COLOR = "#BFD7EA"
ACCENT_COLOR = "#FF9F1C"
TEXT_COLOR = "#333333"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Roboto Slab', serif;
        color: {TEXT_COLOR};
        background-color: #f7f9fc;
    }}
    .title {{
        font-size: 38px;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        text-transform: uppercase;
        margin-bottom: 15px;
        text-align: center;
    }}
    .section-header {{
        font-size: 24px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        color: {PRIMARY_COLOR};
        border-bottom: 3px solid {ACCENT_COLOR};
        padding-bottom: 5px;
    }}
    .caption-text {{
        font-size: 12px;
        color: #666666;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 10px;
    }}
    .card {{
        background: #ffffff;
        padding: 20px 30px;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-radius: 12px;
    }}
    div.stButton > button:first-child {{
        background-color: {ACCENT_COLOR};
        color: white;
        font-weight: 600;
        border-radius: 10px;
        height: 45px;
        width: 100%;
        font-size: 18px;
        transition: background-color 0.3s ease;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #cc7a00;
        color: white;
    }}
    .stTextInput > label, .stNumberInput > label {{
        font-weight: 600;
        color: {PRIMARY_COLOR};
    }}
    /* Adjust expander header font */
    .streamlit-expanderHeader {{
        font-weight: 600 !important;
        color: {PRIMARY_COLOR} !important;
        font-size: 18px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Page Title ---
st.markdown('<div class="title">Company Financial Health & News Sentiment Analyzer</div>', unsafe_allow_html=True)

# --- Financial Inputs Section ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💰 Enter Financial Metrics</div>', unsafe_allow_html=True)
    st.markdown("*(All values should be numerical. Enter percentages as decimals, e.g., 0.1 for 10%)*")

    num_cols_per_row_in_expander = 3
    for category, cols_in_category in FINANCIAL_CATEGORIES.items():
        with st.expander(f"**{category}**", expanded=False):
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
                        key=f"fin_input_{col_name}"
                    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- Sentiment Input Section ---
if sentiment_input_needed:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📰 News Article URL(s) for Sentiment Analysis</div>', unsafe_allow_html=True)

        st.session_state.news_article_urls = st.text_area(
            "Paste News Article URL(s) Here (one per line)",
            value=st.session_state.news_article_urls,
            height=150,
            key="news_article_urls_input"
        )

        st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Button Section ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.button(f"Predict Credit Rating(s)", key="predict_button"):
        # Prepare input data
        current_financial_inputs_df = pd.DataFrame([st.session_state.financial_inputs])

        # Validate inputs before proceeding
        if not all(val is not None for val in st.session_state.financial_inputs.values()):
            st.warning("Please fill in all financial metrics.")
            st.stop()

        # Extract URLs list
        urls_list = [url.strip() for url in st.session_state.news_article_urls.split('\n') if url.strip()] if sentiment_input_needed else []

        # Perform sentiment analysis if needed
        if (selected_model_name == "All Model B (Financial + Sentiment)" or "Model B (Financial + Sentiment)" in selected_model_name):
            if not urls_list:
                st.warning("Please enter news article URL(s) for sentiment analysis.")
                st.stop()
            overall_sentiment_result, detailed_url_results = analyze_multiple_urls_sentiment(urls_list)
            if not overall_sentiment_result or 'category' not in overall_sentiment_result:
                st.warning("Sentiment analysis failed. Please check URLs.")
                st.stop()
        else:
            overall_sentiment_result = None

        # Validate model selection
        if selected_model_name.startswith("---"):
            st.warning("Please select a valid model or comparison option.")
            st.stop()

        # Begin results display
        st.header("3. Prediction Result(s)")

        # Single model prediction
        if selected_model_name in models:
            input_df_for_prediction = current_financial_inputs_df.copy()

            if "Financial Only" in selected_model_name:
                sentiment_features_for_model = {col: 0.0 for col in SENTIMENT_COLS}
                all_inputs = {**st.session_state.financial_inputs, **sentiment_features_for_model}
                input_df_for_prediction = pd.DataFrame([all_inputs])
                required_features_for_model_call = FINANCIAL_COLS
            else:
                sentiment_features_for_model = {col: overall_sentiment_result.get(col, 0.0) for col in SENTIMENT_COLS} if overall_sentiment_result else {col: 0.0 for col in SENTIMENT_COLS}
                all_inputs = {**st.session_state.financial_inputs, **sentiment_features_for_model}
                input_df_for_prediction = pd.DataFrame([all_inputs])
                required_features_for_model_call = ALL_COLS

            input_df_for_prediction = input_df_for_prediction[required_features_for_model_call]

            with st.spinner(f"Predicting with {selected_model_name}..."):
                predicted_rating, probabilities = _predict_single_model(selected_model, selected_scaler, input_df_for_prediction, required_features_for_model_call, label_encoder)

                st.success(f"The predicted credit rating for {st.session_state.company_name} using **{selected_model_name}** is: **{predicted_rating}**")

                with st.expander(f"What is '{predicted_rating}'?"):
                    st.write(f"**{predicted_rating}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating, 'Definition not available.')}")

                st.write("---")
                st.subheader("Prediction Probabilities:")
                prob_df = pd.DataFrame(probabilities.items(), columns=['Rating', 'Probability'])
                prob_df['Probability'] = prob_df['Probability'].astype(float)
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

                st.write("---")
                st.subheader("Key Feature Contributions (Overall Importance):")

                plot_feature_contributions(selected_model, required_features_for_model_call, selected_model_name)

        else:
            st.info("Running multiple models. Results will be displayed below.")

            model_A_names = sorted([name for name in models.keys() if "Model A" in name])
            model_B_names = sorted([name for name in models.keys() if "Model B" in name])

            cols_per_row = 4

            # All Model A
            if model_A_names and (selected_model_name == "All Model A (Financial Only)" or selected_model_name == "All Models (A & B)"):
                st.subheader("All Model A (Financial Only) Predictions:")
                with st.container():
                    for i in range(0, len(model_A_names), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j < len(model_A_names):
                                model_name_to_run = model_A_names[i + j]
                                model = models[model_name_to_run]
                                scaler = scalers[model_name_to_run]

                                zero_sentiment_features = {col: 0.0 for col in SENTIMENT_COLS}
                                all_inputs_for_model_A = {**st.session_state.financial_inputs, **zero_sentiment_features}
                                input_df_for_prediction = pd.DataFrame([all_inputs_for_model_A])

                                features_for_model_call = FINANCIAL_COLS
                                input_df_for_prediction = input_df_for_prediction[features_for_model_call]

                                with cols[j]:
                                    st.markdown(f"**{model_name_to_run}**")
                                    with st.spinner(f"Predicting..."):
                                        predicted_rating, probabilities = _predict_single_model(model, scaler, input_df_for_prediction, features_for_model_call, label_encoder)

                                        if predicted_rating != "Prediction failed.":
                                            st.success(f"Rating: **{predicted_rating}**")
                                            with st.expander(f"What is '{predicted_rating}'?"):
                                                st.write(f"**{predicted_rating}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating, 'Definition not available.')}")

                                            st.markdown("**Probabilities:**")
                                            prob_df = pd.DataFrame(probabilities.items(), columns=['Rating', 'Probability'])
                                            prob_df['Probability'] = prob_df['Probability'].astype(float)
                                            prob_df = prob_df.sort_values(by='Probability', ascending=False)
                                            prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                                            st.dataframe(prob_df, hide_index=True, use_container_width=True)

                                            st.markdown("**Feature Importance:**")
                                            plot_feature_contributions(model, features_for_model_call, model_name_to_run)
                                        else:
                                            st.error(f"Prediction failed.")
                st.markdown("---")

            # All Model B
            if model_B_names and (selected_model_name == "All Model B (Financial + Sentiment)" or selected_model_name == "All Models (A & B)"):
                st.subheader("All Model B (Financial + Sentiment) Predictions:")
                with st.container():
                    for i in range(0, len(model_B_names), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j < len(model_B_names):
                                model_name_to_run = model_B_names[i + j]
                                model = models[model_name_to_run]
                                scaler = scalers[model_name_to_run]

                                all_inputs_for_model_B = {**st.session_state.financial_inputs, **{col: overall_sentiment_result.get(col, 0.0) for col in SENTIMENT_COLS}} if overall_sentiment_result else {}

                                if not all_inputs_for_model_B:
                                    st.error("Missing inputs for Model B.")
                                    continue

                                input_df_for_prediction = pd.DataFrame([all_inputs_for_model_B])
                                features_for_model_call = ALL_COLS
                                input_df_for_prediction = input_df_for_prediction[features_for_model_call]

                                with cols[j]:
                                    st.markdown(f"**{model_name_to_run}**")
                                    with st.spinner(f"Predicting..."):
                                        predicted_rating, probabilities = _predict_single_model(model, scaler, input_df_for_prediction, features_for_model_call, label_encoder)

                                        if predicted_rating != "Prediction failed.":
                                            st.success(f"Rating: **{predicted_rating}**")
                                            with st.expander(f"What is '{predicted_rating}'?"):
                                                st.write(f"**{predicted_rating}:** {CREDIT_RATING_DEFINITIONS.get(predicted_rating, 'Definition not available.')}")

                                            st.markdown("**Probabilities:**")
                                            prob_df = pd.DataFrame(probabilities.items(), columns=['Rating', 'Probability'])
                                            prob_df['Probability'] = prob_df['Probability'].astype(float)
                                            prob_df = prob_df.sort_values(by='Probability', ascending=False)
                                            prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                                            st.dataframe(prob_df, hide_index=True, use_container_width=True)

                                            st.markdown("**Feature Importance:**")
                                            plot_feature_contributions(model, features_for_model_call, model_name_to_run)
                                        else:
                                            st.error(f"Prediction failed.")
                st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)










