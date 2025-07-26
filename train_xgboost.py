import pandas as pd
import numpy as np
import xgboost as xgb # Import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib # For saving/loading scalers, label encoders, and XGBoost models

# Suppress XGBoost deprecation warning for use_label_encoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# --- Configuration and Data Loading ---
try:
    # Assuming 'Finaldata.xlsx' is your Excel file
    df = pd.read_excel("Finaldata.xlsx")
    print(f"Initial data loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'Finaldata.xlsx' not found. Please ensure the file is in the same directory.")
    print("If your file is a CSV, please change pd.read_excel to pd.read_csv.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Target column for credit rating prediction
target_col = 'Rating'

# Financial features (20 features)
financial_cols = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin',
    'returnOnAssets', 'returnOnEquity', 'assetTurnover',
    'fixedAssetTurnover', 'debtRatio', 'effectiveTaxRate',
    'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare',
    'enterpriseValueMultiple', 'payablesTurnover','operatingCashFlowPerShare', 'operatingCashFlowSalesRatio'
]

# Sentiment features (4 features)
sentiment_cols = ['Avg_Positive', 'Avg_Neutral', 'Avg_Negative', 'Avg_Compound']

# Combine all features for Model B
all_features_for_model_B = financial_cols + sentiment_cols

# Define the order of credit ratings for consistent encoding and plotting
RATING_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']

# 🔹 Data Cleaning and Type Conversion
# Convert all selected feature columns to numeric, coercing errors to NaN
print("Converting feature columns to numeric...")
for col in all_features_for_model_B:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Feature column '{col}' not found in DataFrame. Skipping numeric conversion for this column.")

# Drop rows that have missing values in the selected features or the target column
# This ensures that both X_fin and X_all will have complete data for their respective feature sets.
initial_rows = df.shape[0]
df.dropna(subset=all_features_for_model_B + [target_col], inplace=True)
print(f"Dropped {initial_rows - df.shape[0]} rows with missing values in relevant features or target.")
print(f"Data shape after cleaning: {df.shape}")

# Filter out rows where 'Rating' is not in our defined order (if any exist)
df = df[df[target_col].isin(RATING_ORDER)].copy()
print(f"Data shape after filtering for valid ratings: {df.shape}")

if df.empty:
    print("Error: No valid data remaining after cleaning and filtering. Cannot train model.")
    exit()

# Encode the target variable 'Rating' to numerical labels
label_encoder = LabelEncoder()
df['Rating_Encoded'] = label_encoder.fit_transform(df[target_col])
num_classes = len(label_encoder.classes_)
print(f"Encoded credit rating categories: {label_encoder.classes_}")
print(f"Number of classes: {num_classes}")

# Identify classes with less than 2 samples and filter them out
# This prevents the ValueError in train_test_split when using stratify
class_counts = df['Rating_Encoded'].value_counts()
min_samples_per_class = 2 # Minimum required by train_test_split with stratify
classes_to_keep = class_counts[class_counts >= min_samples_per_class].index

if len(classes_to_keep) < num_classes:
    print(f"Warning: Filtering out classes with less than {min_samples_per_class} samples for stratification.")
    initial_rows_before_stratify_filter = df.shape[0]
    df = df[df['Rating_Encoded'].isin(classes_to_keep)].copy()
    print(f"Dropped {initial_rows_before_stratify_filter - df.shape[0]} rows due to low-frequency classes.")
    # Re-encode if classes were removed, to ensure contiguous numerical labels
    label_encoder = LabelEncoder() # Re-initialize to fit on remaining classes
    df['Rating_Encoded'] = label_encoder.fit_transform(df[target_col])
    num_classes = len(label_encoder.classes_)
    print(f"Re-encoded credit rating categories after filtering: {label_encoder.classes_}")
    print(f"Updated number of classes: {num_classes}")


# Reassign features and target variables based on the cleaned DataFrame
y_encoded = df['Rating_Encoded']
X_fin = df[financial_cols] # Features for Model A
X_all = df[all_features_for_model_B] # Features for Model B

# Check if X_fin or X_all are empty AFTER splitting (or before, as originally)
if X_fin.empty or X_all.empty:
    print("Error: Feature DataFrame is empty after cleaning. Cannot perform train-test split.")
    exit()

# ---------- Helper Function for Plotting Accuracy Chart ----------
def plot_accuracy_chart(y_true_encoded, y_pred_encoded, model_label, le):
    """
    Plots actual vs. correctly predicted counts for each credit rating.
    y_true_encoded: True labels (numerical, encoded)
    y_pred_encoded: Predicted labels (numerical, encoded)
    model_label: Label for the plot title
    le: The LabelEncoder used to transform numerical labels back to original strings
    """
    # Convert encoded labels back to original string ratings for plotting
    y_true_str = le.inverse_transform(y_true_encoded)
    y_pred_str = le.inverse_transform(y_pred_encoded)

    comparison_df = pd.DataFrame({
        'Actual': y_true_str,
        'Predicted': y_pred_str
    })
    comparison_df['Match'] = comparison_df['Actual'] == comparison_df['Predicted']

    # Ensure ratings are sorted according to RATING_ORDER for consistent plotting
    # Filter RATING_ORDER to only include classes present in the *current* LabelEncoder
    ratings_for_plot = [r for r in RATING_ORDER if r in le.classes_]

    correct_counts = comparison_df[comparison_df['Match']].groupby('Actual').size().reindex(ratings_for_plot, fill_value=0)
    total_counts = comparison_df.groupby('Actual').size().reindex(ratings_for_plot, fill_value=0)

    x = np.arange(len(ratings_for_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, total_counts, width, label='Total')
    rects2 = ax.bar(x + width/2, correct_counts, width, label='Correctly Predicted')

    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Credit Rating')
    ax.set_title(f'Credit Rating: Actual vs Correctly Predicted ({model_label})')
    ax.set_xticks(x)
    ax.set_xticklabels(ratings_for_plot)
    ax.legend()

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# ------------------------------------
# 🔷 Model A: Financial Only Training (XGBoost)
# ------------------------------------
print("\n--- Training XGBoost Model A (Financial Only) ---")
X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(
    X_fin, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Initialize and fit StandardScaler for financial features
scaler_fin = StandardScaler()
X_train_fin_scaled = scaler_fin.fit_transform(X_train_fin)
X_test_fin_scaled = scaler_fin.transform(X_test_fin)

# Save the scaler for Model A
joblib.dump(scaler_fin, 'yu_pin_scaler_financial.pkl')
print("Scaler for XGBoost Model A (financial only) saved as 'yu_pin_scaler_financial.pkl'")

# Create and train XGBoost Model A
# For multi-class classification, objective='multi:softmax' and num_class are needed.
# eval_metric='mlogloss' is a common metric for multi-class classification.
xgb_model_A = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss', # Use mlogloss for multi-class logloss
    use_label_encoder=False, # Suppress warning for future versions
    random_state=42
)
xgb_model_A.fit(X_train_fin_scaled, y_train_fin)

# Save XGBoost Model A
joblib.dump(xgb_model_A, 'xgb_model_A_financial.pkl')
print("XGBoost Model A (financial only) saved as 'xgb_model_A_financial.pkl'")

# Evaluate Model A
y_pred_A_encoded = xgb_model_A.predict(X_test_fin_scaled)

accuracy_A = accuracy_score(y_test_fin, y_pred_A_encoded)
print("\n--- XGBoost Model A - Financial Only Evaluation: ---")
print(f"Accuracy: {accuracy_A:.4f}")
print("Classification Report:\n", classification_report(y_test_fin, y_pred_A_encoded, target_names=label_encoder.classes_))

# Plot results for Model A
plot_accuracy_chart(y_test_fin, y_pred_A_encoded, "XGBoost Model A (Financial Only)", label_encoder)

# ------------------------------------
# 🔶 Model B: Financial + Sentiment Training (XGBoost)
# ------------------------------------
print("\n--- Training XGBoost Model B (Financial + Sentiment) ---")
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Initialize and fit StandardScaler for all features
scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)

# Save the scaler for Model B
joblib.dump(scaler_all, 'yu_pin_scaler_all.pkl')
print("Scaler for XGBoost Model B (financial + sentiment) saved as 'yu_pin_scaler_all.pkl'")

# Create and train XGBoost Model B
xgb_model_B = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss', # Use mlogloss for multi-class logloss
    use_label_encoder=False, # Suppress warning for future versions
    random_state=42
)
xgb_model_B.fit(X_train_all_scaled, y_train_all)

# Save XGBoost Model B
joblib.dump(xgb_model_B, 'xgb_model_B_financial_sentiment.pkl')
print("XGBoost Model B (financial + sentiment) saved as 'xgb_model_B_financial_sentiment.pkl'")

# Evaluate Model B
y_pred_B_encoded = xgb_model_B.predict(X_test_all_scaled)

accuracy_B = accuracy_score(y_test_all, y_pred_B_encoded)
print("\n--- XGBoost Model B - Financial + Sentiment Evaluation: ---")
print(f"Accuracy: {accuracy_B:.4f}")
print("Classification Report:\n", classification_report(y_test_all, y_pred_B_encoded, target_names=label_encoder.classes_))

# Plot results for Model B
plot_accuracy_chart(y_test_all, y_pred_B_encoded, "XGBoost Model B (Financial + Sentiment)", label_encoder)

# Save the LabelEncoder for decoding predictions in the Streamlit app
joblib.dump(label_encoder, 'rating_label_encoder.pkl')
print("\nLabel Encoder for 'Rating' saved as 'rating_label_encoder.pkl'")

print("\nFinal training and saving complete for both XGBoost models and scalers.")
print("Remember to update your Streamlit app to load these XGBoost models and the label encoder.")
