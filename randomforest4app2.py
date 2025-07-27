import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load and clean your dataset
try:
    df = pd.read_excel("Finaldata.xlsx")
    df.dropna(inplace=True)
except FileNotFoundError:
    print("❌ Finaldata.xlsx not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Define features
target_col = 'Rating'

financial_cols = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin',
    'returnOnAssets', 'returnOnEquity', 'assetTurnover',
    'fixedAssetTurnover', 'debtRatio', 'effectiveTaxRate',
    'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare', 'cashPerShare',
    'enterpriseValueMultiple', 'payablesTurnover','operatingCashFlowPerShare', 'operatingCashFlowSalesRatio'
]

sentiment_cols = ['Avg_Positive', 'Avg_Neutral', 'Avg_Negative', 'Avg_Compound']
all_features_for_model_B = financial_cols + sentiment_cols

# Ensure numeric
for col in all_features_for_model_B:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=all_features_for_model_B + [target_col], inplace=True)

y = df[target_col]
X_fin = df[financial_cols]
X_all = df[all_features_for_model_B]

# Helper: Plot bar chart
def plot_accuracy_chart(y_true, y_pred, label_names, title):
    actual_counts = np.zeros(len(label_names), dtype=int)
    correct_counts = np.zeros(len(label_names), dtype=int)

    for true, pred in zip(y_true, y_pred):
        actual_counts[true] += 1
        if true == pred:
            correct_counts[true] += 1 

    x = np.arange(len(label_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, actual_counts, width, label='Total')
    ax.bar(x + width/2, correct_counts, width, label='Correctly Predicted')

    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Credit Rating')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.legend()

    for i in range(len(label_names)):
        ax.text(x[i] - width/2, actual_counts[i] + 0.2, f'{int(actual_counts[i])}', ha='center', va='bottom')
        ax.text(x[i] + width/2, correct_counts[i] + 0.2, f'{int(correct_counts[i])}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Encode target
label_names = sorted(df[target_col].unique())
label_to_num = {label: i for i, label in enumerate(label_names)}
y_encoded = df[target_col].map(label_to_num).values

# --- Model A ---
print("--- Training Model A (Financial Only) ---")
X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(X_fin, y_encoded, test_size=0.3, random_state=42)

scaler_fin = StandardScaler()
X_train_fin_scaled = scaler_fin.fit_transform(X_train_fin)
X_test_fin_scaled = scaler_fin.transform(X_test_fin)

modelA = RandomForestClassifier(n_estimators=100, random_state=42)
modelA.fit(X_train_fin_scaled, y_train_fin)

y_pred_fin = modelA.predict(X_test_fin_scaled)
accuracy_fin = accuracy_score(y_test_fin, y_pred_fin) * 100

print(f"\n🔹 Model A Accuracy: {accuracy_fin:.2f}%")
print("Classification Report:\n", classification_report(y_test_fin, y_pred_fin))

# Commented out saves
import joblib
joblib.dump(modelA, 'ath_modelA_randomforest.pkl')
joblib.dump(scaler_fin, 'ath_scaler_financial.pkl')
# with open("ath_modelA_accuracy.txt", "w") as f:
#     f.write(f"{accuracy_fin:.2f}%")

plot_accuracy_chart(y_test_fin, y_pred_fin, label_names, "Model A (Financial Only)")

# --- Model B ---
print("\n--- Training Model B (Financial + Sentiment) ---")
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_encoded, test_size=0.3, random_state=42)

scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)

modelB = RandomForestClassifier(n_estimators=100, random_state=42)
modelB.fit(X_train_all_scaled, y_train_all)

y_pred_all = modelB.predict(X_test_all_scaled)
accuracy_all = accuracy_score(y_test_all, y_pred_all) * 100

print(f"\n🔶 Model B Accuracy: {accuracy_all:.2f}%")
print("Classification Report:\n", classification_report(y_test_all, y_pred_all))

# Commented out saves
joblib.dump(modelB, 'ath_modelB_randomforest.pkl')
joblib.dump(scaler_all, 'ath_scaler_all.pkl')
# with open("ath_modelB_accuracy.txt", "w") as f:
#     f.write(f"{accuracy_all:.2f}%")
# pd.DataFrame({ "Model": ["A", "B"], "Accuracy": [accuracy_fin, accuracy_all] }).to_excel("ath_model_accuracy_summary.xlsx", index=False)

plot_accuracy_chart(y_test_all, y_pred_all, label_names, "Model B (Financial + Sentiment)")

print("\n✅ Training done. No files saved. Graphs displayed only.")
