import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score



#load the dataset 
netflix_data = pd.read_csv('Netflix_Feature_Engineered.csv')


# X = features, y = target
y = netflix_data['churned']
x = netflix_data.drop(columns=['churned'])


#Split Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)


#Build and Train Models-----------------------------------------------------------------------------------------------------------------------

#Logistic Regression

LR_model= LogisticRegression()
LR_model.fit(x_train,y_train)
y_pred_LR= LR_model.predict(x_test)

#Random Forest

RF_model = RandomForestClassifier(n_estimators=100,random_state=42)
RF_model.fit(x_train,y_train)
y_pred_RF =RF_model.predict(x_test)

#XGBoost

XGB_model= XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
XGB_model.fit(x_train, y_train)
y_pred_XGB = XGB_model.predict(x_test)


#Evaluate Models

def evaluate_model(y_test, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n")

#Evaluate all models
evaluate_model(y_test, y_pred_LR, "Logistic Regression")
evaluate_model(y_test, y_pred_RF, "Random Forest")
evaluate_model(y_test, y_pred_XGB, "XGBoost")


# Training accuracy
y_train_pred = XGB_model.predict(x_train)
train_acc = accuracy_score(y_train, y_train_pred)

# Test accuracy
y_test_pred = XGB_model.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)



from sklearn.metrics import roc_curve, auc

# ===================== ROC CURVES =====================
# Logistic Regression
y_prob_LR = LR_model.predict_proba(x_test)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_LR)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Random Forest
y_prob_RF = RF_model.predict_proba(x_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_RF)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# XGBoost
y_prob_XGB = XGB_model.predict_proba(x_test)[:,1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_XGB)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC curves together
plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_lr:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_rf:.2f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_auc_xgb:.2f})")

plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.show()


# ===================== CONFUSION MATRIX =====================
models = {
    "Logistic Regression": y_pred_LR,
    "Random Forest": y_pred_RF,
    "XGBoost": y_pred_XGB
}

for name, y_pred in models.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"])
    plt.title(f"{name} - Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


# ===================== FEATURE IMPORTANCE =====================


# Logistic Regression Feature Importance
log_feat_importances = pd.Series(np.abs(LR_model.coef_[0]), index=x.columns)

plt.figure(figsize=(8,6))
log_feat_importances.nlargest(10).plot(kind="barh", color="skyblue")
plt.title("Top 10 Feature Importances - Logistic Regression")
plt.xlabel("Coefficient (absolute value)")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()

# Random Forest Feature Importance

feat_importances = pd.Series(RF_model.feature_importances_, index=x.columns)
plt.figure(figsize=(8,6))
feat_importances.nlargest(10).plot(kind="barh")
plt.title("Top 10 Feature Importances - Random Forest")
plt.show()

# XGBoost Feature Importance
plot_importance(XGB_model, max_num_features=10, importance_type="weight")
plt.title("Top 10 Feature Importances - XGBoost")
plt.show()   


