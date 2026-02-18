# ============================================
# TASK 1: Load and Understand the Dataset
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv(r"C:\Users\HP\22481A0533\AI_Hiring_Prediction_System\data\AI-Based Hiring Prediction System.csv")

print(df.head())
print(df.tail())
print(df.sample(5))

"""
Explanation:
This dataset contains both numerical and textual resume data.
Target variable: Recruiter Decision
"""

# ============================================
# TASK 2: Basic Data Inspection
# ============================================

print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.dtypes)

print(df["Recruiter Decision"].value_counts())
print(df.describe())

"""
Why inspection matters:
It helps identify data types, imbalance, missing values,
and ensures correct preprocessing before model training.
"""

# ============================================
# TASK 3: Data Cleaning and Preprocessing
# ============================================

# Fill missing text fields
df["Skills"] = df["Skills"].fillna("")
df["Certifications"] = df["Certifications"].fillna("")
df["Job Role"] = df["Job Role"].fillna("")

df.drop(["Resume_ID", "Name"], axis=1, inplace=True)

df["Recruiter Decision"] = df["Recruiter Decision"].map({
    "Hire": 1,
    "Reject": 0
})

print(df.isnull().sum())

"""
Missing values:
If present, we would use mean/median for numerical data
and mode or 'Unknown' for categorical/text data.
"""

# ============================================
# TASK 4: Text Feature Engineering
# ============================================

df["combined_text"] = (
    df["Skills"] + " " +
    df["Certifications"] + " " +
    df["Job Role"]
)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join([c for c in text if c.isalnum() or c == " "])
    text = " ".join(text.split())
    return text


df["combined_text"] = (
    df["Skills"] + " " +
    df["Certifications"] + " " +
    df["Job Role"]
)

df["combined_text"] = df["combined_text"].apply(clean_text)


"""
Why text cleaning:
Removes noise, ensures consistent tokens,
and improves vectorization quality.
"""

# ============================================
# TASK 5: Convert Text to Numerical Features
# ============================================

tfidf = TfidfVectorizer(max_features=300)
X_text = tfidf.fit_transform(df["combined_text"])

"""
ML models need numbers, not raw text.
TF-IDF measures word importance across documents.
"""

# ============================================
# TASK 6: Encode Categorical Variables
# ============================================

le = LabelEncoder()
df["Education"] = le.fit_transform(df["Education"])

"""
Label Encoding: assigns numbers to categories.
One-Hot Encoding: creates binary columns.
"""

# ============================================
# TASK 7: Feature and Target Separation
# ============================================

X_numeric = df[[
    "Experience (Years)",
    "Salary Expectation ($)",
    "Projects Count",
    "Education"
]]

y = df["Recruiter Decision"]

"""
Target should never be in features to avoid data leakage.
"""

# ============================================
# TASK 8: Train–Test Split
# ============================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

X_train_num, X_test_num = train_test_split(
    X_numeric, test_size=0.2, random_state=42
)

"""
Train-test split checks generalization.
Overfitting: model memorizes training data.
"""

# ============================================
# TASK 9: Feature Scaling
# ============================================

scaler = StandardScaler()

X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

"""
Scaling is needed for distance-based models
like SVM and KNN.
"""

# ============================================
# TASK 10: Model Training
# ============================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train_num_scaled, y_train)
    preds = model.predict(X_test_num_scaled)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc

    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

"""
Model explanations:
Logistic Regression – linear decision boundary
Random Forest – ensemble of decision trees
SVM – maximizes margin
KNN – distance-based classification
"""

# ============================================
# TASK 11: Model Comparison
# ============================================

comparison_df = pd.DataFrame({
    "Model Name": accuracies.keys(),
    "Accuracy": accuracies.values()
})

print(comparison_df)

"""
Best model:
Typically Random Forest or SVM due to mixed features
and non-linear relationships.
"""

# ============================================
# TASK 12: Pipeline + GridSearch (Advanced)
# ============================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=300)),
    ("clf", LogisticRegression(max_iter=1000))
])

param_grid = {
    "clf__C": [0.1, 1, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(df["combined_text"], y)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

"""
Pipelines ensure consistent preprocessing
during training and inference.
"""

# ============================================
# TASK 13: Hiring Prediction Function
# ============================================

def predict_hiring(skills, experience, education, certs, projects, salary):
    text = clean_text(skills + " " + certs)
    text_vec = tfidf.transform([text])

    edu_encoded = le.transform([education])[0]

    num_data = scaler.transform([[
        experience,
        salary,
        projects,
        edu_encoded
    ]])

    model = RandomForestClassifier()
    model.fit(X_train_num_scaled, y_train)

    prediction = model.predict(num_data)[0]
    probability = model.predict_proba(num_data)[0][prediction]

    return ("Hire" if prediction == 1 else "Reject"), probability

# ============================================
# TASK 14: Final Conclusion
# ============================================

"""
Conclusion:
We analyzed resume data, performed preprocessing,
engineered text features, trained multiple ML models,
and built a hiring prediction system.
This mirrors real-world HR automation systems
used for resume screening and decision support.
"""
