# 🎓 Nirmaan 2.1 — Cutoff Analytics & ML Predictor

An interactive, production-ready Streamlit web application for exploratory data analysis (EDA) and predictive machine learning on college admission cutoff data.

---

## 🚀 Features

- **Executive KPI Dashboard**: Instant summary of total records, universities, programmes, year range, and average cutoff scores.
- **Exploratory Data Analysis (EDA)**:
  - Top 10 Universities by Cutoff Records
  - Cutoff Figure Kinds (Score vs Rank) distribution
  - Distribution of Score Figures with KDE (Kernel Density Estimate)
  - Log-scale boxplot of figures across admission years
  - Pearson correlation metrics and summary statistics
- **Dynamic Data Filtering**:
  - Filter by University, Figure Kind, Year range, and category
  - Built-in search engine for Programmes and Colleges
  - CSV export for filtered results
- **Machine Learning Regression Models**:
  - Compares **Linear Regression**, **Random Forest Regressor**, and **K-Nearest Neighbors (KNN)**
  - Real-time evaluation with **Root Mean Squared Error (RMSE)** and **R² Score**
  - Displays mathematical formulation ($y = mx + c$) with slope & intercept
  - Model comparison visualization
- **Cutoff Predictor Tool**:
  - Forecasts expected cutoff figures for target admission years (e.g., 2027, 2028).

---

## 📦 Local Installation & Setup

1. **Clone or navigate to the repository directory:**
   ```bash
   cd "Nirmaan 2.1"
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   *(Or alternatively: `streamlit run untitled7.py`)*

4. Open your web browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push this folder to your GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository and branch (`main`).
5. Set **Main file path** to `app.py`.
6. Click **Deploy!**

Streamlit Cloud will automatically detect `requirements.txt` and `.streamlit/config.toml` and deploy the application.
