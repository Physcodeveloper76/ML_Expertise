# -*- coding: utf-8 -*-
"""
Nirmaan 2.1 - University Cutoff Analytics & ML Prediction App
Instant Visualizations & Model Evaluation on Data Feed
"""

import sys
import warnings
from pathlib import Path

# Suppress sklearn and display warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import streamlit.runtime as rt

# Auto-runner: If executed via 'python app.py' directly, automatically launch Streamlit
if not rt.exists():
    from streamlit.web import cli as stcli
    print("[*] Launching Streamlit Web App...")
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nirmaan | Cutoff Analytics & Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    .score-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .score-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .score-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1E293B;
    }
    .score-sub {
        font-size: 0.8rem;
        color: #059669;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "cutoffs.csv"

# ---------------------------------------------------------
# Smart Data Loader & Column Detection
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_csv_data(file_path_or_buffer) -> pd.DataFrame:
    try:
        if isinstance(file_path_or_buffer, (str, Path)):
            df = pd.read_csv(file_path_or_buffer)
        else:
            df = pd.read_csv(file_path_or_buffer)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

def detect_column(df: pd.DataFrame, candidates: list):
    """Finds best matching column name case-insensitively."""
    cols_lower = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower().strip() in cols_lower:
            return cols_lower[candidate.lower().strip()]
    return None

# ---------------------------------------------------------
# ML Training Function (Cached)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Training machine learning models...")
def train_and_evaluate_models(df: pd.DataFrame, year_col: str, figure_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Trains Linear Regression, Random Forest, and KNN Regressor.
    Returns metrics, predictions, and model instances.
    """
    cleaned = df.dropna(subset=[year_col, figure_col]).copy()
    cleaned[year_col] = pd.to_numeric(cleaned[year_col], errors='coerce')
    cleaned[figure_col] = pd.to_numeric(cleaned[figure_col], errors='coerce')
    cleaned = cleaned.dropna(subset=[year_col, figure_col])

    if len(cleaned) < 10 or cleaned[year_col].nunique() < 2:
        return None

    X = cleaned[[year_col]]
    y = cleaned[figure_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # 3. KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    models_dict = {
        "Linear Regression": (lr, lr_pred),
        "Random Forest": (rf, rf_pred),
        "KNN": (knn, knn_pred)
    }

    metrics = {}
    best_model_name = None
    best_r2 = -float("inf")

    for name, (model, preds) in models_dict.items():
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))
        metrics[name] = {
            "RMSE": rmse,
            "R2": r2,
            "model": model,
            "preds": preds
        }
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    # Pearson Correlation
    pearson_corr = float(cleaned[year_col].corr(cleaned[figure_col], method='pearson'))

    return {
        "metrics": metrics,
        "best_model_name": best_model_name,
        "lr_slope": float(lr.coef_[0]),
        "lr_intercept": float(lr.intercept_),
        "pearson_corr": pearson_corr,
        "total_trained_samples": len(cleaned),
        "y_test": y_test,
        "X_test": X_test,
        "year_col": year_col,
        "figure_col": figure_col
    }

# ---------------------------------------------------------
# Sidebar: Data Feeding & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=60)
    st.title("Feed Data")
    st.write("Upload any dataset to instantly generate all visualisations and evaluation scores.")

    uploaded_file = st.file_uploader("📥 Upload CSV file", type=["csv"], help="Upload your college cutoffs or admissions dataset")

    use_default = False
    if uploaded_file is None:
        if DEFAULT_DATA_PATH.exists():
            st.info("Using default `cutoffs.csv` (84,210 rows).")
            use_default = True
        else:
            st.warning("Please upload a CSV file to proceed.")
            st.stop()

    st.markdown("---")
    st.subheader("Data Settings")

# Load raw dataset
if uploaded_file is not None:
    raw_df = load_csv_data(uploaded_file)
    data_source_label = f"Uploaded File: {uploaded_file.name}"
elif use_default:
    raw_df = load_csv_data(DEFAULT_DATA_PATH)
    data_source_label = "Default Dataset: cutoffs.csv"
else:
    st.stop()

if raw_df.empty:
    st.error("The loaded dataset is empty. Please feed a valid dataset.")
    st.stop()

# ---------------------------------------------------------
# Column Mapping & Smart Detection
# ---------------------------------------------------------
detected_year = detect_column(raw_df, ['year', 'admission_year', 'exam_year', 'session'])
detected_figure = detect_column(raw_df, ['figure', 'cutoff', 'score', 'marks', 'closing_score', 'closing_rank', 'rank'])
detected_univ = detect_column(raw_df, ['university', 'college', 'institution', 'university_name'])
detected_kind = detect_column(raw_df, ['figure_kind', 'kind', 'type', 'cutoff_type'])

with st.sidebar:
    year_col = st.selectbox(
        "Year Column",
        options=raw_df.columns,
        index=raw_df.columns.get_loc(detected_year) if (detected_year and detected_year in raw_df.columns) else 0
    )
    figure_col = st.selectbox(
        "Figure / Cutoff Column",
        options=raw_df.columns,
        index=raw_df.columns.get_loc(detected_figure) if (detected_figure and detected_figure in raw_df.columns) else 0
    )
    univ_col = st.selectbox(
        "University / Group Column",
        options=["(None)"] + list(raw_df.columns),
        index=(list(raw_df.columns).index(detected_univ) + 1) if (detected_univ and detected_univ in raw_df.columns) else 0
    )
    kind_col = st.selectbox(
        "Figure Kind Column (Optional)",
        options=["(None)"] + list(raw_df.columns),
        index=(list(raw_df.columns).index(detected_kind) + 1) if (detected_kind and detected_kind in raw_df.columns) else 0
    )

    st.markdown("---")
    st.subheader("Filters")

    # Filter by University if available
    filtered_df = raw_df.copy()
    if univ_col != "(None)":
        univ_options = sorted([str(x) for x in raw_df[univ_col].dropna().unique()])
        selected_univs = st.multiselect("Filter University", options=univ_options, default=[])
        if selected_univs:
            filtered_df = filtered_df[filtered_df[univ_col].astype(str).isin(selected_univs)]

    # Filter by Figure Kind if available
    if kind_col != "(None)":
        kind_options = sorted([str(x) for x in raw_df[kind_col].dropna().unique()])
        selected_kinds = st.multiselect("Filter Figure Kind", options=kind_options, default=kind_options)
        if selected_kinds:
            filtered_df = filtered_df[filtered_df[kind_col].astype(str).isin(selected_kinds)]

    st.caption(f"Active Records: **{len(filtered_df):,}** / {len(raw_df):,}")

# Ensure numeric conversions
filtered_df[year_col] = pd.to_numeric(filtered_df[year_col], errors='coerce')
filtered_df[figure_col] = pd.to_numeric(filtered_df[figure_col], errors='coerce')

# ---------------------------------------------------------
# Instant Model Training & Score Calculation
# ---------------------------------------------------------
if kind_col != "(None)" and 'score' in filtered_df[kind_col].astype(str).str.lower().values:
    ml_train_df = filtered_df[filtered_df[kind_col].astype(str).str.lower() == 'score']
    if (ml_train_df[figure_col] <= 1000).sum() > 10:
        ml_train_df = ml_train_df[(ml_train_df[figure_col] > 0) & (ml_train_df[figure_col] <= 1000)]
else:
    ml_train_df = filtered_df

ml_results = train_and_evaluate_models(ml_train_df, year_col, figure_col)

# ---------------------------------------------------------
# Top Header & Summary KPIs
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 Nirmaan Cutoff Analytics & Predictor</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Data Source: <b>{data_source_label}</b> | Instantly generating all visualisations and model scores upon data ingestion</div>', unsafe_allow_html=True)

# Top KPI row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Rows", f"{len(filtered_df):,}")
with k2:
    if univ_col != "(None)":
        st.metric("Institutions", f"{filtered_df[univ_col].nunique():,}")
    else:
        st.metric("Columns", f"{len(filtered_df.columns)}")
with k3:
    yr_min = filtered_df[year_col].dropna().min()
    yr_max = filtered_df[year_col].dropna().max()
    st.metric("Year Span", f"{int(yr_min)} – {int(yr_max)}" if pd.notna(yr_min) and pd.notna(yr_max) else "N/A")
with k4:
    mean_val = filtered_df[figure_col].dropna().mean()
    st.metric("Mean Cutoff", f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A")
with k5:
    if ml_results:
        st.metric("Pearson Correlation", f"{ml_results['pearson_corr']:.4f}")
    else:
        st.metric("Correlation", "N/A")

st.markdown("---")

# ---------------------------------------------------------
# SECTION 1: INSTANT SCORES & METRICS DASHBOARD
# ---------------------------------------------------------
st.subheader("🎯 Model Performance & Evaluation Scores")
st.write("Scores calculated automatically whenever data is fed into the system:")

if ml_results:
    m = ml_results["metrics"]
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-title">Linear Regression</div>
            <div class="score-value">RMSE: {m['Linear Regression']['RMSE']:.2f}</div>
            <div class="score-sub">R² Score: {m['Linear Regression']['R2']:.5f}</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-title">Random Forest</div>
            <div class="score-value">RMSE: {m['Random Forest']['RMSE']:.2f}</div>
            <div class="score-sub">R² Score: {m['Random Forest']['R2']:.5f}</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-title">KNN Regressor (k=5)</div>
            <div class="score-value">RMSE: {m['KNN']['RMSE']:.2f}</div>
            <div class="score-sub">R² Score: {m['KNN']['R2']:.5f}</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col4:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-title">Pearson Correlation (r)</div>
            <div class="score-value">{ml_results['pearson_corr']:.4f}</div>
            <div class="score-sub">Slope: {ml_results['lr_slope']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Linear Regression Formula Callout
    st.info(f"""
    📐 **Linear Regression Formula:** `Cutoff = ({ml_results['lr_slope']:.4f} × Year) + ({ml_results['lr_intercept']:.4f})`  
    • **Slope (m):** `{ml_results['lr_slope']:.4f}` | **Intercept (c):** `{ml_results['lr_intercept']:.4f}` | **Best Model:** `{ml_results['best_model_name']}`
    """)
else:
    st.warning("⚠️ Insufficient variance or data points in Year/Figure columns to compute regression scores. Ensure Year and Figure columns contain valid numeric entries.")

st.markdown("---")

# ---------------------------------------------------------
# SECTION 2: ALL VISUALIZATIONS GRID
# ---------------------------------------------------------
st.subheader("📊 Visualizations Dashboard")
sns.set_theme(style="whitegrid")

viz_row1_col1, viz_row1_col2 = st.columns(2)

# Visualization 1: Top Universities / Groups
with viz_row1_col1:
    st.markdown("#### 1. Top 10 Universities by Cutoff Records")
    if univ_col != "(None)" and univ_col in filtered_df.columns:
        top_u = filtered_df[univ_col].dropna().value_counts().head(10)
        if not top_u.empty:
            fig1, ax1 = plt.subplots(figsize=(7, 4.2))
            sns.barplot(
                y=top_u.index.astype(str),
                x=top_u.values,
                ax=ax1,
                hue=top_u.index.astype(str),
                palette="viridis",
                legend=False
            )
            ax1.set_xlabel("Number of Cutoff Entries")
            ax1.set_ylabel("University")
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("No university records available.")
    else:
        st.info("Select a University column from sidebar to view top records.")

# Visualization 2: Distribution of Figure Kinds
with viz_row1_col2:
    st.markdown("#### 2. Distribution of Figure Kinds (Score vs Rank)")
    if kind_col != "(None)" and kind_col in filtered_df.columns:
        kinds_data = filtered_df.dropna(subset=[kind_col])
        if not kinds_data.empty:
            fig2, ax2 = plt.subplots(figsize=(7, 4.2))
            sns.countplot(
                data=kinds_data,
                x=kind_col,
                ax=ax2,
                hue=kind_col,
                palette="pastel",
                legend=False
            )
            ax2.set_xlabel("Figure Kind")
            ax2.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("No figure kind data.")
    else:
        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        sns.countplot(
            data=filtered_df.dropna(subset=[year_col]),
            x=year_col,
            ax=ax2,
            palette="pastel"
        )
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Records Count")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

viz_row2_col1, viz_row2_col2 = st.columns(2)

# Visualization 3: Distribution of Figure Values (Scores with KDE)
with viz_row2_col1:
    st.markdown("#### 3. Distribution of Figure Values (Histogram & KDE)")
    fig_vals = filtered_df[figure_col].dropna()
    if (fig_vals <= 1000).sum() > 20:
        fig_vals = fig_vals[(fig_vals > 0) & (fig_vals <= 1000)]
    
    if not fig_vals.empty:
        fig3, ax3 = plt.subplots(figsize=(7, 4.2))
        sns.histplot(x=fig_vals, kde=True, ax=ax3, color='skyblue', bins=40)
        ax3.set_xlabel("Cutoff Value")
        ax3.set_ylabel("Density / Frequency")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("No figure values to display.")

# Visualization 4: Boxplot of Figures by Year (Log Scale)
with viz_row2_col2:
    st.markdown("#### 4. Boxplot of Cutoff Figures by Year (Log Scale)")
    box_data = filtered_df.dropna(subset=[year_col, figure_col])
    if not box_data.empty and box_data[year_col].nunique() > 0:
        fig4, ax4 = plt.subplots(figsize=(7, 4.2))
        sns.boxplot(
            data=box_data,
            x=year_col,
            y=figure_col,
            ax=ax4,
            hue=year_col,
            palette="Set2",
            legend=False
        )
        if (box_data[figure_col] > 0).all():
            ax4.set_yscale('log')
            ax4.set_ylabel("Figure (Log Scale)")
        else:
            ax4.set_ylabel("Figure")
        ax4.set_xlabel("Year")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
    else:
        st.info("Insufficient data for boxplot.")

# Visualization 5: Model Fit & Trend Comparison
if ml_results:
    st.markdown("---")
    st.markdown("#### 5. Machine Learning Models Trend & Prediction Curve")
    fig5, ax5 = plt.subplots(figsize=(10, 4.2))
    
    years_to_plot = sorted(list(filtered_df[year_col].dropna().unique()))
    if len(years_to_plot) > 0:
        future_x = [int(y) for y in list(years_to_plot) + [int(max(years_to_plot)) + 1, int(max(years_to_plot)) + 2]]
        future_df = pd.DataFrame({year_col: future_x})
        
        lr_line = ml_results['metrics']['Linear Regression']['model'].predict(future_df)
        rf_line = ml_results['metrics']['Random Forest']['model'].predict(future_df)
        knn_line = ml_results['metrics']['KNN']['model'].predict(future_df)

        ax5.plot(future_x, lr_line, marker='o', label='Linear Regression', color='#4F46E5', linewidth=2)
        ax5.plot(future_x, rf_line, marker='s', label='Random Forest', color='#10B981', linewidth=2)
        ax5.plot(future_x, knn_line, marker='^', label='KNN', color='#F59E0B', linewidth=2)
        
        actual_means = filtered_df.groupby(year_col)[figure_col].mean()
        ax5.scatter(actual_means.index, actual_means.values, color='crimson', s=90, zorder=5, label='Actual Year Mean')

        ax5.set_xlabel("Year")
        ax5.set_ylabel("Predicted Cutoff Figure")
        ax5.set_title("Fitted Regression Models with Projection")
        ax5.legend()
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

st.markdown("---")

# ---------------------------------------------------------
# SECTION 3: INTERACTIVE TABS (PREDICTOR & DATA EXPLORER)
# ---------------------------------------------------------
tab_pred, tab_exp = st.tabs(["🔮 Cutoff Predictor Tool", "🔍 Browse & Export Data"])

with tab_pred:
    st.subheader("Predict Future Cutoff for Any Year")
    p1, p2 = st.columns([1, 1])
    with p1:
        next_yr_val = int(filtered_df[year_col].dropna().max() + 1) if not filtered_df[year_col].dropna().empty else 2027
        input_yr = st.number_input("Enter Target Year", min_value=2020, max_value=2040, value=next_yr_val, step=1)
        selected_model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "KNN"])
        
    with p2:
        if ml_results:
            chosen_model = ml_results['metrics'][selected_model_type]['model']
            pred_df = pd.DataFrame({year_col: [input_yr]})
            predicted_figure = chosen_model.predict(pred_df)[0]
            st.success(f"### Predicted Cutoff: **{predicted_figure:.2f}**")
            st.caption(f"Calculated via **{selected_model_type}** for Year **{input_yr}**")
            if selected_model_type == "Linear Regression":
                st.code(f"Formula: ({ml_results['lr_slope']:.4f} × {input_yr}) + ({ml_results['lr_intercept']:.4f}) = {predicted_figure:.2f}")
        else:
            st.info("Trainable model unavailable.")

with tab_exp:
    st.subheader("Data Explorer")
    search_txt = st.text_input("Search records...", placeholder="Search any term...")
    display_df = filtered_df.copy()
    if search_txt:
        mask = False
        for c in display_df.columns:
            mask = mask | display_df[c].astype(str).str.contains(search_txt, case=False, na=False)
        display_df = display_df[mask]

    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.write(f"Showing **{min(len(display_df), 500):,}** of **{len(display_df):,}** records")
    with col_r:
        csv_bytes = display_df.head(5000).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV", data=csv_bytes, file_name="nirmaan_cutoffs_export.csv", mime="text/csv")

    try:
        st.dataframe(display_df.head(500), width="stretch")
    except TypeError:
        st.dataframe(display_df.head(500))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "Nirmaan 2.1 • Instant Visualizations & ML Scores Engine"
    "</div>",
    unsafe_allow_html=True
)
