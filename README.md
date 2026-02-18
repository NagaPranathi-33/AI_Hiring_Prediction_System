```md
# AI-Based Hiring Prediction System

An end-to-end Machine Learning project that predicts whether a candidate will be **Hired** or **Rejected** based on resume data.  
This project simulates a real-world AI-powered resume screening system used in HR analytics.

---

## 📌 Project Objective

To build a machine learning model that predicts recruiter decisions using structured resume information such as:
- Skills
- Experience
- Education
- Certifications
- Job Role
- Salary Expectation
- Project Experience

The target variable is **Recruiter Decision**:
- `Hire` → `1`
- `Reject` → `0`

---

## 📂 Project Structure

```

AI_Hiring_Prediction_System/
│
├── data/
│   └── AI-Based Hiring Prediction System.csv
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/   (ignored)

````

---

## 📊 Dataset Description

The dataset contains **1,000 synthetic resumes** with the following features:

| Column Name | Description |
|------------|-------------|
| Resume_ID | Unique identifier |
| Name | Candidate name |
| Skills | Technical skills |
| Experience (Years) | Total experience |
| Education | Highest qualification |
| Certifications | Relevant certifications |
| Job Role | Applied job role |
| Salary Expectation ($) | Expected salary |
| Projects Count | Number of projects |
| AI Score (0–100) | Precomputed score (not used) |
| Recruiter Decision | Target variable |

> **Note:** `Resume_ID`, `Name`, and `AI Score` are dropped to prevent data leakage.

---

## 🛠️ Technologies & Packages Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TF-IDF (NLP)

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
````

### 2. Activate Virtual Environment (Windows)

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

---

## 🧠 Machine Learning Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Text feature engineering (Skills, Certifications, Job Role)
4. TF-IDF vectorization
5. Encoding categorical features
6. Feature scaling
7. Train–test split
8. Model training and evaluation
9. Model comparison
10. Optional pipeline and hyperparameter tuning

---

## 🤖 Models Used

| Model                  | Description             |
| ---------------------- | ----------------------- |
| Logistic Regression    | Linear classification   |
| Random Forest          | Ensemble-based learning |
| Support Vector Machine | Margin-based classifier |
| K-Nearest Neighbors    | Distance-based model    |

---

## 📈 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 96.5%    |
| Random Forest       | 96.5%    |
| SVM                 | 95.0%    |
| KNN                 | 94.0%    |

> Logistic Regression and Random Forest performed best due to strong correlations in numerical features.

---

## 🔬 Advanced Features

* **TF-IDF + Pipeline**
* **GridSearchCV for hyperparameter tuning**
* Clean handling of missing textual data
* Production-style preprocessing steps

---

## 🧪 Hiring Prediction Function

The system includes a function that:

* Accepts candidate details as input
* Returns hiring decision (Hire/Reject)
* Outputs probability score

This simulates a real-world AI resume screening tool.

---

## 📌 Key Learnings

* Importance of data inspection before modeling
* Handling missing text data in NLP pipelines
* Feature scaling and model selection
* Avoiding data leakage
* Building reproducible ML workflows

---

## 🌍 Real-World Applications

* Automated resume screening
* HR analytics systems
* Talent acquisition platforms
* Decision-support tools for recruiters

---

## 🏁 Conclusion

This project demonstrates a complete machine learning pipeline for resume screening, combining data preprocessing, NLP techniques, and multiple classification models. It closely mirrors real-world HR automation systems and highlights best practices in applied machine learning.

---

## 📜 License

This project is for educational purposes.

```
