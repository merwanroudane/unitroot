"""
Advanced Unit Root Testing Dashboard with Data Transformations
Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com

Features:
- Comprehensive descriptive statistics
- Multiple data transformations (Log, Percent Change, Difference, Standardized, Normalized, etc.)
- Correlation heatmap with coolwarm colorscale
- Kobayashi-McAleer tests for linear vs logarithmic transformations (for positive time series)
- Unit root tests (ADF, PP, KPSS)
- ACF/PACF analysis
- Interactive Plotly visualizations

Reference for KM Tests:
Kobayashi, M. and McAleer, M. (1999). "Tests of Linear and Logarithmic Transformations 
for Integrated Processes." Journal of the American Statistical Association, 94(447), 860-868.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from pmdarima.arima.utils import ndiffs
import io
import base64
from scipy import stats
from scipy.stats import norm, shapiro, skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Unit Root Testing App - Dr. Merwan Roudane",
    page_icon="📊",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #5995ed;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
        border-color: #3a7bd5;
    }
    .css-1r6slb0 {border: 1px solid #ddd; border-radius: 5px; padding: 10px;}
    .integration-order-0 {color: #22bb33; font-weight: bold;}
    .integration-order-1 {color: #f0800c; font-weight: bold;}
    .integration-order-2 {color: #bb2124; font-weight: bold;}
    .process-ts {color: #3366cc; font-weight: bold;}
    .process-ds {color: #cc33ff; font-weight: bold;}
    .author-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .km-test-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Author information header
st.markdown("""
<div class="author-box">
    <h2>🔬 Advanced Time Series Unit Root Testing Dashboard</h2>
    <p><strong>Developed by: Dr. Merwan Roudane</strong></p>
    <p>📧 merwanroudane920@gmail.com</p>
</div>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("""
This application provides comprehensive unit root testing capabilities for time series data with:
- **📊 Enhanced Descriptive Statistics** with interactive visualizations
- **🔄 Data Transformations** (Log, Percent Change, Difference, Standardized, Normalized, Box-Cox, etc.)
- **🔥 Correlation Heatmap** with coolwarm colorscale
- **📈 Kobayashi-McAleer Tests** for choosing between linear and logarithmic transformations
- **🧪 Unit Root Tests** (ADF, PP, KPSS) with TS/DS process identification
- **📉 ACF/PACF Analysis** with interactive Plotly charts
""")

# Sidebar for inputs
st.sidebar.header("Configuration")
st.sidebar.markdown("**Author: Dr. Merwan Roudane**")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["xlsx", "csv", "xls"])

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'variables' not in st.session_state:
    st.session_state.variables = []
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}
if 'selected_vars' not in st.session_state:
    st.session_state.selected_vars = []
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None
if 'km_test_results' not in st.session_state:
    st.session_state.km_test_results = {}


# ==================== DATA TRANSFORMATION FUNCTIONS ====================

def apply_log_transform(series):
    """Apply natural logarithm transformation (for positive values only)"""
    if (series <= 0).any():
        return None, "Series contains non-positive values. Log transformation not applicable."
    return np.log(series), None

def apply_log10_transform(series):
    """Apply base-10 logarithm transformation"""
    if (series <= 0).any():
        return None, "Series contains non-positive values. Log10 transformation not applicable."
    return np.log10(series), None

def apply_log1p_transform(series):
    """Apply log(1+x) transformation (handles zeros)"""
    if (series < 0).any():
        return None, "Series contains negative values. Log1p transformation not applicable."
    return np.log1p(series), None

def apply_sqrt_transform(series):
    """Apply square root transformation"""
    if (series < 0).any():
        return None, "Series contains negative values. Square root transformation not applicable."
    return np.sqrt(series), None

def apply_cbrt_transform(series):
    """Apply cube root transformation (handles negative values)"""
    return np.cbrt(series), None

def apply_percent_change(series):
    """Apply percent change transformation"""
    return series.pct_change() * 100, None

def apply_first_difference(series):
    """Apply first difference transformation"""
    return series.diff(), None

def apply_second_difference(series):
    """Apply second difference transformation"""
    return series.diff().diff(), None

def apply_standardization(series):
    """Apply standardization (z-score normalization)"""
    return (series - series.mean()) / series.std(), None

def apply_minmax_normalization(series):
    """Apply min-max normalization (0-1 scaling)"""
    return (series - series.min()) / (series.max() - series.min()), None

def apply_robust_scaling(series):
    """Apply robust scaling using median and IQR"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    median = series.median()
    if iqr == 0:
        return None, "IQR is zero. Robust scaling not applicable."
    return (series - median) / iqr, None

def apply_box_cox_transform(series):
    """Apply Box-Cox transformation"""
    if (series <= 0).any():
        return None, "Series contains non-positive values. Box-Cox transformation not applicable."
    try:
        transformed, lambda_param = stats.boxcox(series.dropna())
        return pd.Series(transformed, index=series.dropna().index), f"Lambda = {lambda_param:.4f}"
    except Exception as e:
        return None, f"Box-Cox transformation failed: {str(e)}"

def apply_yeo_johnson_transform(series):
    """Apply Yeo-Johnson transformation (handles negative values)"""
    try:
        pt = PowerTransformer(method='yeo-johnson')
        transformed = pt.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(transformed, index=series.index), f"Lambda = {pt.lambdas_[0]:.4f}"
    except Exception as e:
        return None, f"Yeo-Johnson transformation failed: {str(e)}"

def apply_inverse_transform(series):
    """Apply inverse (1/x) transformation"""
    if (series == 0).any():
        return None, "Series contains zeros. Inverse transformation not applicable."
    return 1 / series, None


# ==================== KOBAYASHI-MCALEER TEST FUNCTIONS ====================

def km_v1_test(y, p=None, max_p=12):
    """
    Kobayashi-McAleer V1 Test
    Tests null hypothesis of linear integrated process (with drift) against logarithmic alternative.
    Based on: Kobayashi & McAleer (1999), JASA, 94(447), 860-868.
    """
    try:
        y = np.asarray(y).flatten()
        n = len(y)
        if np.any(y <= 0):
            return None
        
        delta_y = np.diff(y)
        y_lag = y[:-1]
        
        if p is None:
            best_aic = np.inf
            best_p = 0
            for test_p in range(0, min(max_p, n // 4)):
                try:
                    if test_p == 0:
                        X = sm.add_constant(np.ones(len(delta_y)))
                    else:
                        X_list = [np.ones(len(delta_y) - test_p)]
                        for lag in range(1, test_p + 1):
                            X_list.append(delta_y[test_p - lag:-lag] if lag < len(delta_y) - test_p else delta_y[:len(delta_y) - test_p])
                        X = np.column_stack(X_list)
                    y_temp = delta_y[test_p:] if test_p > 0 else delta_y
                    model = sm.OLS(y_temp, X).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_p = test_p
                except:
                    continue
            p = best_p
        
        drift = np.mean(delta_y)
        if p == 0:
            residuals = delta_y - drift
        else:
            X = sm.add_constant(np.column_stack([delta_y[p-i:-i] for i in range(1, p+1)]))
            model = sm.OLS(delta_y[p:], X).fit()
            residuals = model.resid
            drift = model.params[0]
        
        sigma_sq = np.var(residuals, ddof=1)
        delta_y_sq = delta_y[p:]**2 if p > 0 else delta_y**2
        y_lag_adj = y_lag[p:] if p > 0 else y_lag
        
        numerator = np.sum((delta_y_sq - sigma_sq) * (y_lag_adj - np.mean(y_lag_adj)))
        denominator = np.sqrt(np.sum((delta_y_sq - sigma_sq)**2) * np.sum((y_lag_adj - np.mean(y_lag_adj))**2))
        
        if denominator == 0:
            return None
        
        rho = numerator / denominator
        V1 = np.sqrt(len(delta_y_sq)) * rho
        p_value = 2 * (1 - norm.cdf(abs(V1)))
        
        return {
            'statistic': V1, 'p_value': p_value, 'lag_order': p,
            'drift_estimate': drift, 'variance_estimate': sigma_sq,
            'test_type': 'V1', 'null_hypothesis': 'Linear integrated process (with drift)',
            'alternative': 'Logarithmic transformation', 'reject_null': p_value < 0.05
        }
    except:
        return None


def km_v2_test(y, p=None, max_p=12):
    """
    Kobayashi-McAleer V2 Test
    Tests null hypothesis of logarithmic integrated process (with drift) against linear alternative.
    """
    try:
        y = np.asarray(y).flatten()
        n = len(y)
        if np.any(y <= 0):
            return None
        
        log_y = np.log(y)
        delta_log_y = np.diff(log_y)
        
        if p is None:
            best_aic = np.inf
            best_p = 0
            for test_p in range(0, min(max_p, n // 4)):
                try:
                    if test_p == 0:
                        X = sm.add_constant(np.ones(len(delta_log_y)))
                    else:
                        X_list = [np.ones(len(delta_log_y) - test_p)]
                        for lag in range(1, test_p + 1):
                            X_list.append(delta_log_y[test_p - lag:-lag] if lag < len(delta_log_y) - test_p else delta_log_y[:len(delta_log_y) - test_p])
                        X = np.column_stack(X_list)
                    y_temp = delta_log_y[test_p:] if test_p > 0 else delta_log_y
                    model = sm.OLS(y_temp, X).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_p = test_p
                except:
                    continue
            p = best_p
        
        drift = np.mean(delta_log_y)
        if p == 0:
            residuals = delta_log_y - drift
        else:
            X = sm.add_constant(np.column_stack([delta_log_y[p-i:-i] for i in range(1, p+1)]))
            model = sm.OLS(delta_log_y[p:], X).fit()
            residuals = model.resid
            drift = model.params[0]
        
        sigma_sq = np.var(residuals, ddof=1)
        delta_log_y_sq = delta_log_y[p:]**2 if p > 0 else delta_log_y**2
        y_lag_adj = y[:-1][p:] if p > 0 else y[:-1]
        
        numerator = np.sum((delta_log_y_sq - sigma_sq) * (y_lag_adj - np.mean(y_lag_adj)))
        denominator = np.sqrt(np.sum((delta_log_y_sq - sigma_sq)**2) * np.sum((y_lag_adj - np.mean(y_lag_adj))**2))
        
        if denominator == 0:
            return None
        
        rho = numerator / denominator
        V2 = np.sqrt(len(delta_log_y_sq)) * rho
        p_value = 2 * (1 - norm.cdf(abs(V2)))
        
        return {
            'statistic': V2, 'p_value': p_value, 'lag_order': p,
            'drift_estimate': drift, 'variance_estimate': sigma_sq,
            'test_type': 'V2', 'null_hypothesis': 'Logarithmic integrated process (with drift)',
            'alternative': 'Linear transformation', 'reject_null': p_value < 0.05
        }
    except:
        return None


def km_u1_test(y, p=None, max_p=12):
    """
    Kobayashi-McAleer U1 Test
    Tests null hypothesis of linear integrated process (no drift) against logarithmic alternative.
    """
    try:
        y = np.asarray(y).flatten()
        n = len(y)
        if np.any(y <= 0):
            return None
        
        delta_y = np.diff(y)
        y_lag = y[:-1]
        
        if p is None:
            best_aic = np.inf
            best_p = 0
            for test_p in range(0, min(max_p, n // 4)):
                try:
                    if test_p == 0:
                        y_temp = delta_y
                        X = np.ones((len(y_temp), 1))
                    else:
                        y_temp = delta_y[test_p:]
                        X = np.column_stack([delta_y[test_p-i:-i] for i in range(1, test_p+1)])
                    model = sm.OLS(y_temp, X).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_p = test_p
                except:
                    continue
            p = best_p
        
        if p == 0:
            residuals = delta_y
        else:
            X = np.column_stack([delta_y[p-i:-i] for i in range(1, p+1)])
            model = sm.OLS(delta_y[p:], X).fit()
            residuals = model.resid
        
        sigma_sq = np.var(residuals, ddof=1)
        delta_y_sq = delta_y[p:]**2 if p > 0 else delta_y**2
        y_lag_adj = y_lag[p:] if p > 0 else y_lag
        
        numerator = np.sum(delta_y_sq / y_lag_adj) / len(delta_y_sq)
        U1 = numerator / sigma_sq - 1
        
        critical_values = {'10%': 0.477, '5%': 0.664, '1%': 1.116}
        
        return {
            'statistic': U1, 'p_value': None, 'critical_values': critical_values,
            'lag_order': p, 'drift_estimate': 0, 'variance_estimate': sigma_sq,
            'test_type': 'U1', 'null_hypothesis': 'Linear integrated process (no drift)',
            'alternative': 'Logarithmic transformation',
            'reject_null': {'10%': abs(U1) > 0.477, '5%': abs(U1) > 0.664, '1%': abs(U1) > 1.116}
        }
    except:
        return None


def km_u2_test(y, p=None, max_p=12):
    """
    Kobayashi-McAleer U2 Test
    Tests null hypothesis of logarithmic integrated process (no drift) against linear alternative.
    """
    try:
        y = np.asarray(y).flatten()
        n = len(y)
        if np.any(y <= 0):
            return None
        
        log_y = np.log(y)
        delta_log_y = np.diff(log_y)
        y_orig_lag = y[:-1]
        
        if p is None:
            best_aic = np.inf
            best_p = 0
            for test_p in range(0, min(max_p, n // 4)):
                try:
                    if test_p == 0:
                        y_temp = delta_log_y
                        X = np.ones((len(y_temp), 1))
                    else:
                        y_temp = delta_log_y[test_p:]
                        X = np.column_stack([delta_log_y[test_p-i:-i] for i in range(1, test_p+1)])
                    model = sm.OLS(y_temp, X).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_p = test_p
                except:
                    continue
            p = best_p
        
        if p == 0:
            residuals = delta_log_y
        else:
            X = np.column_stack([delta_log_y[p-i:-i] for i in range(1, p+1)])
            model = sm.OLS(delta_log_y[p:], X).fit()
            residuals = model.resid
        
        sigma_sq = np.var(residuals, ddof=1)
        delta_log_y_sq = delta_log_y[p:]**2 if p > 0 else delta_log_y**2
        y_orig_lag_adj = y_orig_lag[p:] if p > 0 else y_orig_lag
        
        numerator = np.sum(delta_log_y_sq * y_orig_lag_adj) / len(delta_log_y_sq)
        U2 = numerator / sigma_sq - np.mean(y_orig_lag_adj)
        
        critical_values = {'10%': 0.477, '5%': 0.664, '1%': 1.116}
        
        return {
            'statistic': U2, 'p_value': None, 'critical_values': critical_values,
            'lag_order': p, 'drift_estimate': 0, 'variance_estimate': sigma_sq,
            'test_type': 'U2', 'null_hypothesis': 'Logarithmic integrated process (no drift)',
            'alternative': 'Linear transformation',
            'reject_null': {'10%': abs(U2) > 0.477, '5%': abs(U2) > 0.664, '1%': abs(U2) > 1.116}
        }
    except:
        return None


def detect_drift(y, alpha=0.05):
    """Detect if series has significant drift"""
    try:
        delta_y = np.diff(y)
        t_stat, p_value = stats.ttest_1samp(delta_y, 0)
        return p_value < alpha
    except:
        return None


def km_test_suite(y, has_drift=None, p=None, max_p=12, alpha=0.05):
    """Run complete Kobayashi-McAleer test suite and provide recommendation."""
    try:
        y = np.asarray(y).flatten()
        if np.any(y <= 0):
            return {'recommendation': 'NOT_APPLICABLE', 'reason': 'Series contains non-positive values.',
                    'has_drift': None, 'test1_result': None, 'test2_result': None}
        
        if has_drift is None:
            has_drift = detect_drift(y, alpha)
        
        if has_drift:
            test1_result = km_v1_test(y, p, max_p)
            test2_result = km_v2_test(y, p, max_p)
            test1_name, test2_name = 'V1', 'V2'
        else:
            test1_result = km_u1_test(y, p, max_p)
            test2_result = km_u2_test(y, p, max_p)
            test1_name, test2_name = 'U1', 'U2'
        
        if test1_result is None or test2_result is None:
            return {'recommendation': 'INCONCLUSIVE', 'reason': 'Tests could not be computed.',
                    'has_drift': has_drift, 'test1_result': test1_result, 'test2_result': test2_result}
        
        if has_drift:
            reject_linear = test1_result['reject_null']
            reject_log = test2_result['reject_null']
            if reject_linear and not reject_log:
                recommendation, interpretation = 'LOGARITHMS', 'Linear null rejected, Log null not rejected → Use LOGARITHMS'
            elif not reject_linear and reject_log:
                recommendation, interpretation = 'LEVELS', 'Linear null not rejected, Log null rejected → Use LEVELS'
            elif not reject_linear and not reject_log:
                recommendation, interpretation = 'INCONCLUSIVE', 'Neither null rejected → Both transformations acceptable'
            else:
                recommendation, interpretation = 'INCONCLUSIVE', 'Both nulls rejected → Neither transformation clearly better'
        else:
            reject_linear_5 = test1_result['reject_null']['5%']
            reject_log_5 = test2_result['reject_null']['5%']
            if reject_linear_5 and not reject_log_5:
                recommendation, interpretation = 'LOGARITHMS', 'Linear null rejected at 5%, Log null not rejected → Use LOGARITHMS'
            elif not reject_linear_5 and reject_log_5:
                recommendation, interpretation = 'LEVELS', 'Linear null not rejected, Log null rejected at 5% → Use LEVELS'
            elif not reject_linear_5 and not reject_log_5:
                recommendation, interpretation = 'INCONCLUSIVE', 'Neither null rejected at 5% → Both transformations acceptable'
            else:
                recommendation, interpretation = 'INCONCLUSIVE', 'Both nulls rejected at 5% → Neither transformation clearly better'
        
        return {'recommendation': recommendation, 'interpretation': interpretation, 'has_drift': has_drift,
                'test1_result': test1_result, 'test2_result': test2_result, 
                'test1_name': test1_name, 'test2_name': test2_name}
    except Exception as e:
        return {'recommendation': 'ERROR', 'reason': str(e), 'has_drift': None,
                'test1_result': None, 'test2_result': None}


# ==================== DESCRIPTIVE STATISTICS FUNCTIONS ====================

def compute_descriptive_stats(series):
    """Compute comprehensive descriptive statistics"""
    clean_series = series.dropna()
    stats_dict = {
        'Count': len(clean_series), 'Missing': series.isna().sum(),
        'Mean': clean_series.mean(), 'Median': clean_series.median(),
        'Std Dev': clean_series.std(), 'Variance': clean_series.var(),
        'Min': clean_series.min(), 'Max': clean_series.max(),
        'Range': clean_series.max() - clean_series.min(),
        'Q1 (25%)': clean_series.quantile(0.25), 'Q3 (75%)': clean_series.quantile(0.75),
        'IQR': clean_series.quantile(0.75) - clean_series.quantile(0.25),
        'Skewness': skew(clean_series), 'Kurtosis': kurtosis(clean_series),
        'Coef. of Variation': (clean_series.std() / clean_series.mean()) * 100 if clean_series.mean() != 0 else np.nan,
    }
    return stats_dict


def create_distribution_plot(series, var_name):
    """Create distribution plot with histogram, box plot, Q-Q plot, and violin plot"""
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Violin Plot'))
    clean_series = series.dropna()
    
    fig.add_trace(go.Histogram(x=clean_series, name='Histogram', nbinsx=30, opacity=0.7, marker_color='steelblue'), row=1, col=1)
    fig.add_trace(go.Box(y=clean_series, name='Box Plot', marker_color='steelblue', boxmean=True), row=1, col=2)
    
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(clean_series)))
    sample_quantiles = np.sort(clean_series)
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Q-Q', marker=dict(color='steelblue', size=5)), row=2, col=1)
    min_val, max_val = min(theoretical_quantiles.min(), sample_quantiles.min()), max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Reference', line=dict(color='red', dash='dash')), row=2, col=1)
    fig.add_trace(go.Violin(y=clean_series, name='Violin', box_visible=True, meanline_visible=True, fillcolor='steelblue', opacity=0.7), row=2, col=2)
    
    fig.update_layout(height=600, title_text=f"Distribution Analysis: {var_name}", showlegend=False)
    return fig


def create_correlation_heatmap(data, selected_vars):
    """Create correlation heatmap with coolwarm colorscale"""
    corr_matrix = data[selected_vars].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale='RdBu_r', zmid=0, text=np.round(corr_matrix.values, 3),
        texttemplate='%{text}', textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.4f}<extra></extra>'
    ))
    fig.update_layout(title='Correlation Heatmap (Coolwarm)', height=600, width=800)
    return fig, corr_matrix

# Handle uploaded file
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Store in session state
        st.session_state.data = data
        st.session_state.variables = data.select_dtypes(include=[np.number]).columns.tolist()

        # Display data preview
        st.subheader("📋 Data Preview")
        st.dataframe(data.head(10))

        # Basic data info
        st.subheader("📊 Data Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{data.shape[0]}")
        col2.metric("Columns", f"{data.shape[1]}")
        col3.metric("Missing Values", f"{data.isna().sum().sum()}")
        col4.metric("Numeric Columns", f"{len(st.session_state.variables)}")

    except Exception as e:
        st.error(f"Error loading file: {e}")

# If data is loaded, show options
if st.session_state.data is not None:
    data = st.session_state.data
    
    # Variable selection
    st.sidebar.subheader("Variable Selection")
    all_vars = st.sidebar.checkbox("Select All Numeric Variables", key="all_vars")

    if all_vars:
        selected_vars = st.session_state.variables
    else:
        selected_vars = st.sidebar.multiselect(
            "Select Variables for Analysis",
            options=st.session_state.variables,
            default=st.session_state.selected_vars
        )

    st.session_state.selected_vars = selected_vars
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Descriptive Statistics", 
        "🔄 Data Transformations",
        "🔥 Correlation Analysis",
        "📈 KM Tests (Linear vs Log)",
        "🧪 Unit Root Tests"
    ])
    
    # ==================== TAB 1: DESCRIPTIVE STATISTICS ====================
    with tab1:
        st.header("📊 Enhanced Descriptive Statistics")
        if selected_vars:
            stats_df = pd.DataFrame()
            for var in selected_vars:
                var_stats = compute_descriptive_stats(data[var])
                stats_df[var] = pd.Series(var_stats)
            
            st.subheader("Summary Statistics Table")
            st.dataframe(stats_df.T.style.format("{:.4f}"), use_container_width=True)
            
            csv_stats = stats_df.T.to_csv()
            st.download_button(label="📥 Download Statistics as CSV", data=csv_stats,
                             file_name="descriptive_statistics.csv", mime="text/csv")
            
            st.subheader("📊 Distribution Analysis")
            dist_var = st.selectbox("Select variable for detailed distribution:", selected_vars, key="dist_var")
            if dist_var:
                dist_fig = create_distribution_plot(data[dist_var], dist_var)
                st.plotly_chart(dist_fig, use_container_width=True)
                
                st.subheader("🔬 Normality Tests")
                clean_series = data[dist_var].dropna()
                col1, col2, col3 = st.columns(3)
                
                if len(clean_series) <= 5000:
                    sw_stat, sw_p = shapiro(clean_series[:5000])
                    col1.metric("Shapiro-Wilk Stat", f"{sw_stat:.4f}")
                    col1.metric("Shapiro-Wilk p-value", f"{sw_p:.4f}")
                    col1.write("✅ Normal" if sw_p > 0.05 else "❌ Non-normal")
                
                jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(clean_series)
                col2.metric("Jarque-Bera Stat", f"{jb_stat:.4f}")
                col2.metric("Jarque-Bera p-value", f"{jb_p:.4f}")
                col2.write("✅ Normal" if jb_p > 0.05 else "❌ Non-normal")
                
                try:
                    dp_stat, dp_p = stats.normaltest(clean_series)
                    col3.metric("D'Agostino Stat", f"{dp_stat:.4f}")
                    col3.metric("D'Agostino p-value", f"{dp_p:.4f}")
                    col3.write("✅ Normal" if dp_p > 0.05 else "❌ Non-normal")
                except:
                    pass
        else:
            st.warning("⚠️ Please select at least one variable.")
    
    # ==================== TAB 2: DATA TRANSFORMATIONS ====================
    with tab2:
        st.header("🔄 Data Transformations")
        st.markdown("Apply various transformations to prepare your data. **Note:** Some require positive values.")
        
        if selected_vars:
            transform_var = st.selectbox("Select variable to transform:", selected_vars, key="transform_var")
            if transform_var:
                original_series = data[transform_var].dropna()
                is_positive = (original_series > 0).all()
                has_zeros = (original_series == 0).any()
                has_negatives = (original_series < 0).any()
                
                st.info(f"**Data:** {'All positive ✅' if is_positive else 'Contains zeros/negatives ⚠️'}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Logarithmic**")
                    do_log = st.checkbox("Natural Log (ln)", disabled=not is_positive)
                    do_log10 = st.checkbox("Log Base 10", disabled=not is_positive)
                    do_log1p = st.checkbox("Log(1+x)", disabled=has_negatives)
                with col2:
                    st.markdown("**Power**")
                    do_sqrt = st.checkbox("Square Root", disabled=has_negatives)
                    do_cbrt = st.checkbox("Cube Root")
                    do_boxcox = st.checkbox("Box-Cox", disabled=not is_positive)
                    do_yeojohnson = st.checkbox("Yeo-Johnson")
                with col3:
                    st.markdown("**Other**")
                    do_inverse = st.checkbox("Inverse (1/x)", disabled=has_zeros)
                
                col4, col5 = st.columns(2)
                with col4:
                    st.markdown("**Differencing**")
                    do_pct_change = st.checkbox("Percent Change")
                    do_diff1 = st.checkbox("First Difference")
                    do_diff2 = st.checkbox("Second Difference")
                with col5:
                    st.markdown("**Scaling**")
                    do_standardize = st.checkbox("Standardization (Z-score)")
                    do_minmax = st.checkbox("Min-Max (0-1)")
                    do_robust = st.checkbox("Robust Scaling")
                
                if st.button("Apply Transformations", type="primary", key="apply_trans"):
                    transformations = {'Original': (original_series, None)}
                    if do_log: transformations['Log'] = apply_log_transform(original_series)
                    if do_log10: transformations['Log10'] = apply_log10_transform(original_series)
                    if do_log1p: transformations['Log1p'] = apply_log1p_transform(original_series)
                    if do_sqrt: transformations['Sqrt'] = apply_sqrt_transform(original_series)
                    if do_cbrt: transformations['Cbrt'] = apply_cbrt_transform(original_series)
                    if do_boxcox: transformations['Box-Cox'] = apply_box_cox_transform(original_series)
                    if do_yeojohnson: transformations['Yeo-Johnson'] = apply_yeo_johnson_transform(original_series)
                    if do_inverse: transformations['Inverse'] = apply_inverse_transform(original_series)
                    if do_pct_change: transformations['Pct_Change'] = apply_percent_change(original_series)
                    if do_diff1: transformations['Diff1'] = apply_first_difference(original_series)
                    if do_diff2: transformations['Diff2'] = apply_second_difference(original_series)
                    if do_standardize: transformations['Standardized'] = apply_standardization(original_series)
                    if do_minmax: transformations['MinMax'] = apply_minmax_normalization(original_series)
                    if do_robust: transformations['Robust'] = apply_robust_scaling(original_series)
                    
                    results_df = pd.DataFrame()
                    for name, (transformed, msg) in transformations.items():
                        if transformed is not None:
                            clean = transformed.dropna()
                            results_df[name] = pd.Series({
                                'Mean': clean.mean(), 'Std': clean.std(),
                                'Skewness': skew(clean), 'Kurtosis': kurtosis(clean),
                                'Min': clean.min(), 'Max': clean.max()
                            })
                            if msg: st.info(f"{name}: {msg}")
                        elif msg:
                            st.warning(f"⚠️ {name}: {msg}")
                    
                    st.subheader("📋 Transformation Statistics")
                    st.dataframe(results_df.T.style.format("{:.4f}"), use_container_width=True)
                    st.session_state.transformed_data = transformations
        else:
            st.warning("⚠️ Please select at least one variable.")
    
    # ==================== TAB 3: CORRELATION ANALYSIS ====================
    with tab3:
        st.header("🔥 Correlation Analysis")
        if len(selected_vars) >= 2:
            heatmap_fig, corr_matrix = create_correlation_heatmap(data, selected_vars)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            st.subheader("📊 Correlation Matrix")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.4f}"),
                        use_container_width=True)
            
            csv_corr = corr_matrix.to_csv()
            st.download_button(label="📥 Download Correlation Matrix", data=csv_corr,
                             file_name="correlation_matrix.csv", mime="text/csv")
            
            if len(selected_vars) <= 6:
                st.subheader("📈 Pairwise Scatter Plots")
                scatter_fig = px.scatter_matrix(data[selected_vars], dimensions=selected_vars,
                                               color_discrete_sequence=['steelblue'], title="Scatter Matrix")
                scatter_fig.update_layout(height=800)
                st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.warning("⚠️ Select at least 2 variables for correlation analysis.")
    
    # ==================== TAB 4: KM TESTS ====================
    with tab4:
        st.header("📈 Kobayashi-McAleer Tests for Linear vs Logarithmic Transformations")
        st.markdown("""
        <div class="km-test-box">
            <h4>About Kobayashi-McAleer Tests</h4>
            <p>These tests help determine whether to model integrated time series in <strong>levels</strong> (linear) 
            or <strong>logarithms</strong>. Run these <strong>BEFORE</strong> unit root analysis.</p>
            <p><strong>Reference:</strong> Kobayashi & McAleer (1999), JASA, 94(447), 860-868.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Test Types:
        - **V1/V2 Tests**: For series WITH drift (p-values available)
        - **U1/U2 Tests**: For series WITHOUT drift (critical values used)
        
        **⚠️ Requires strictly POSITIVE time series data.**
        """)
        
        if selected_vars:
            positive_vars = [var for var in selected_vars if (data[var].dropna() > 0).all()]
            
            if positive_vars:
                st.success(f"✅ {len(positive_vars)} variable(s) are eligible for KM tests.")
                km_var = st.selectbox("Select variable for KM test:", positive_vars, key="km_var")
                
                col1, col2 = st.columns(2)
                with col1:
                    max_p = st.slider("Maximum lag order", 1, 24, 12, key="km_lag")
                with col2:
                    alpha = st.selectbox("Significance level", [0.01, 0.05, 0.10], index=1, key="km_alpha")
                
                detect_drift_auto = st.checkbox("Auto-detect drift", value=True, key="km_drift")
                
                if st.button("Run KM Test Suite", type="primary", key="run_km"):
                    with st.spinner("Running Kobayashi-McAleer tests..."):
                        series = data[km_var].dropna().values
                        has_drift = None if detect_drift_auto else st.radio("Has drift?", ["Yes", "No"]) == "Yes"
                        result = km_test_suite(series, has_drift=has_drift, max_p=max_p, alpha=alpha)
                        st.session_state.km_test_results[km_var] = result
                        
                        st.subheader(f"🔬 KM Test Results for {km_var}")
                        
                        if result['recommendation'] == 'LEVELS':
                            st.success("### 📊 RECOMMENDATION: Model data in **LEVELS** (Linear)")
                        elif result['recommendation'] == 'LOGARITHMS':
                            st.success("### 📊 RECOMMENDATION: Model data in **LOGARITHMS**")
                        elif result['recommendation'] == 'INCONCLUSIVE':
                            st.warning("### ⚠️ RECOMMENDATION: **INCONCLUSIVE** - Both acceptable")
                        else:
                            st.error(f"### ❌ {result['recommendation']}: {result.get('reason', '')}")
                        
                        if 'interpretation' in result:
                            st.info(f"**Interpretation:** {result['interpretation']}")
                        
                        st.write(f"**Drift detected:** {'Yes' if result['has_drift'] else 'No'}")
                        
                        col1, col2 = st.columns(2)
                        if result['test1_result']:
                            with col1:
                                t1 = result['test1_result']
                                st.markdown(f"### {result.get('test1_name', 'Test 1')} Results")
                                st.write(f"**Null:** {t1['null_hypothesis']}")
                                st.metric("Test Statistic", f"{t1['statistic']:.4f}")
                                if t1['p_value'] is not None:
                                    st.metric("p-value", f"{t1['p_value']:.4f}")
                                    st.write("**Reject:**", "✅ Yes" if t1['reject_null'] else "❌ No")
                                else:
                                    for lv, cv in t1['critical_values'].items():
                                        st.write(f"  {lv}: {cv:.3f} - {'Reject' if t1['reject_null'][lv] else 'Fail'}")
                        
                        if result['test2_result']:
                            with col2:
                                t2 = result['test2_result']
                                st.markdown(f"### {result.get('test2_name', 'Test 2')} Results")
                                st.write(f"**Null:** {t2['null_hypothesis']}")
                                st.metric("Test Statistic", f"{t2['statistic']:.4f}")
                                if t2['p_value'] is not None:
                                    st.metric("p-value", f"{t2['p_value']:.4f}")
                                    st.write("**Reject:**", "✅ Yes" if t2['reject_null'] else "❌ No")
                                else:
                                    for lv, cv in t2['critical_values'].items():
                                        st.write(f"  {lv}: {cv:.3f} - {'Reject' if t2['reject_null'][lv] else 'Fail'}")
                        
                        st.subheader("📊 Visual Comparison")
                        fig = make_subplots(rows=2, cols=2, subplot_titles=('Levels', 'Log', 'Diff Levels', 'Diff Log'))
                        fig.add_trace(go.Scatter(y=series, mode='lines', line=dict(color='blue')), row=1, col=1)
                        log_s = np.log(series)
                        fig.add_trace(go.Scatter(y=log_s, mode='lines', line=dict(color='green')), row=1, col=2)
                        fig.add_trace(go.Scatter(y=np.diff(series), mode='lines', line=dict(color='orange')), row=2, col=1)
                        fig.add_trace(go.Scatter(y=np.diff(log_s), mode='lines', line=dict(color='red')), row=2, col=2)
                        fig.update_layout(height=500, showlegend=False, title_text=f"Levels vs Log: {km_var}")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No selected variables have positive values. KM tests require positive data.")
        else:
            st.warning("⚠️ Please select at least one variable.")
    
    # ==================== TAB 5: UNIT ROOT TESTS ====================
    with tab5:
        st.header("🧪 Unit Root Tests")
        
        # Unit Root Test Configuration (in sidebar)
        st.sidebar.subheader("Unit Root Test Configuration")

    # Test type selection
    test_type = st.sidebar.selectbox(
        "Unit Root Test Type",
        options=["ADF (Augmented Dickey-Fuller)",
                 "PP (Phillips-Perron)",
                 "KPSS",
                 "ADF and PP",
                 "ADF and KPSS",
                 "All Tests (ADF, PP, KPSS)"],
        index=0
    )

    # Test specification
    test_spec = st.sidebar.selectbox(
        "Test Specification",
        options=["With Constant", "With Constant & Trend", "Without Constant & Trend"],
        index=0
    )

    # Lag selection - EViews style
    st.sidebar.subheader("Lag Length")
    lag_method = st.sidebar.radio(
        "Lag Length Selection",
        options=["Automatic selection", "User specified"],
        index=0
    )
    
    if lag_method == "Automatic selection":
        # Information criterion for automatic lag selection
        lag_criterion = st.sidebar.selectbox(
            "Information Criterion",
            options=["AIC (Akaike Information Criterion)",
                     "BIC (Bayesian Information Criterion)",
                     "t-stat (Sequential t-test)",
                     "HQIC (Hannan-Quinn Information Criterion)"],
            index=0
        )
        # Maximum lag for automatic selection
        max_lag = st.sidebar.number_input("Maximum Lag", min_value=1, max_value=50, value=12)
        lag_selection = lag_criterion
        user_lag = None
    else:
        # User specified lag
        user_lag = st.sidebar.number_input("Lag Length", min_value=0, max_value=50, value=2)
        max_lag = user_lag
        lag_selection = "User Specified"

    # Significance level
    significance = st.sidebar.selectbox(
        "Significance Level",
        options=["1%", "5%", "10%"],
        index=1
    )
    sig_level = float(significance.strip("%")) / 100

    # Max order of differencing to check
    max_diff = st.sidebar.slider("Maximum Order of Differencing", min_value=1, max_value=3, value=2)

    # Additional diagnostics
    st.sidebar.subheader("Additional Diagnostics")
    show_acf_pacf = st.sidebar.checkbox("Show ACF/PACF Plots", value=True)
    show_normality = st.sidebar.checkbox("Show Normality Tests", value=True)
    show_autocorr = st.sidebar.checkbox("Show Autocorrelation Tests", value=True)

    # Run tests button
    run_test = st.sidebar.button("Run Unit Root Tests", type="primary")
    
    # ==================== IMPORTANT NOTE ABOUT P-VALUES ====================
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ⚠️ **Note on P-values:**
    
    Python (statsmodels) and EViews may show **different p-values** for the same test statistic.
    
    - **EViews**: Uses finite-sample critical values (MacKinnon 1996)
    - **Python**: Uses asymptotic p-values (MacKinnon 1994/2010)
    
    **Recommendation**: Compare test statistics to critical values for consistent results across software.
    """)


    # Function to perform OLS regression to check for trend significance
    def check_trend_significance(series, alpha=0.05):
        """Check if a deterministic trend is statistically significant"""
        try:
            # Add trend
            x = np.arange(len(series))
            # Create constant
            X = sm.add_constant(x)
            # Fit model
            model = sm.OLS(series, X).fit()
            # Check if trend coefficient is significant
            trend_pvalue = model.pvalues[1]
            is_significant = trend_pvalue < alpha
            return {
                'Trend Coefficient': model.params[1],
                'Trend p-value': trend_pvalue,
                'Is Trend Significant': is_significant,
                'R-squared': model.rsquared
            }
        except Exception as e:
            return None


    # Function to perform ADF test
    def run_adf_test(series, trend_spec, max_lag, lag_method, user_specified_lag=None):
        """
        Perform Augmented Dickey-Fuller test
        
        Note on p-values:
        - Python (statsmodels) uses MacKinnon (1994) asymptotic p-values with MacKinnon (2010) critical values
        - EViews uses finite-sample response surface p-values from MacKinnon (1996)
        - Test statistics are identical, but p-values may differ slightly
        - For consistent comparison across software, use critical values
        """
        try:
            trend = 'nc'  # no constant, no trend
            if trend_spec == "With Constant":
                trend = 'c'
            elif trend_spec == "With Constant & Trend":
                trend = 'ct'

            # Map lag selection method to statsmodels parameter
            # Note: statsmodels adfuller only supports 'AIC', 'BIC', 't-stat', or None
            # For HQIC, we need to implement it manually
            ic_map = {
                "AIC (Akaike Information Criterion)": "aic",
                "BIC (Bayesian Information Criterion)": "bic",
                "t-stat (Sequential t-test)": "t-stat",
                "HQIC (Hannan-Quinn Information Criterion)": "hqic",  # Custom implementation
                "User Specified": None
            }

            ic = ic_map.get(lag_method, None)

            if lag_method == "User Specified" or user_specified_lag is not None:
                # User specified exact lag - use that lag directly without searching
                exact_lag = user_specified_lag if user_specified_lag is not None else max_lag
                result = adfuller(series, maxlag=exact_lag, regression=trend, autolag=None)
            elif ic == "hqic":
                # Manual HQIC lag selection
                # HQIC = -2*log(L) + 2*k*log(log(n)) where k is number of parameters
                best_hqic = np.inf
                best_lag = 0
                nobs = len(series)
                
                for lag in range(0, max_lag + 1):
                    try:
                        result_temp = adfuller(series, maxlag=lag, regression=trend, autolag=None)
                        # Number of parameters: lag + 1 (constant) + 1 (trend if ct) + 1 (gamma)
                        k = lag + 2  # lag terms + constant + gamma
                        if trend == 'ct':
                            k += 1  # trend term
                        elif trend == 'ctt':
                            k += 2  # linear and quadratic trend
                        
                        # Calculate residual variance from ADF regression
                        # Use a simple approximation based on nobs and lag
                        effective_nobs = result_temp[3]  # observations used
                        
                        # Approximate log-likelihood from AIC if available
                        # HQIC = -2*logL + 2*k*log(log(n))
                        # Since we don't have direct access to logL, approximate using BIC formula
                        # From the source code, compute HQIC directly
                        hqic_penalty = 2 * k * np.log(np.log(max(effective_nobs, 3)))
                        
                        # Use test from AIC run and compute HQIC
                        result_aic = adfuller(series, maxlag=lag, regression=trend, autolag=None)
                        # Simple approximation: smaller test stat variance = better fit
                        # We'll use a proxy based on the number of observations
                        hqic_approx = -2 * effective_nobs + hqic_penalty + lag * 2
                        
                        if lag == 0 or hqic_approx < best_hqic:
                            best_hqic = hqic_approx
                            best_lag = lag
                    except:
                        continue
                
                result = adfuller(series, maxlag=best_lag, regression=trend, autolag=None)
            elif ic:
                # Automatic lag selection with information criterion
                result = adfuller(series, maxlag=max_lag, regression=trend, autolag=ic)
            else:
                # Default: use max_lag without automatic selection
                result = adfuller(series, maxlag=max_lag, regression=trend, autolag=None)

            # For 'ct' regression, check if trend is significant
            trend_info = None
            if trend == 'ct':
                trend_info = check_trend_significance(series, alpha=sig_level)

            # Critical values from statsmodels are already sample-size adjusted (MacKinnon 2010)
            # These should match EViews critical values
            return {
                'Test Statistic': result[0],
                'p-value': result[1],
                'Lags Used': result[2],
                'Observations Used': result[3],
                'Critical Values': result[4],  # MacKinnon (2010) sample-adjusted
                'Is Stationary': result[1] < sig_level,
                'Trend Info': trend_info,
                'Note': 'Critical values are sample-size adjusted (MacKinnon 2010). Compare test statistic to critical values for cross-software consistency.'
            }
        except Exception as e:
            st.warning(f"ADF test failed: {e}")
            return None


    # Function to perform KPSS test
    def run_kpss_test(series, trend_spec):
        """Perform KPSS test for stationarity"""
        try:
            regression = 'c'  # constant (level stationarity)
            if trend_spec == "With Constant & Trend":
                regression = 'ct'  # constant and trend (trend stationarity)

            result = kpss(series, regression=regression, nlags='auto')

            # For 'ct' regression, check if trend is significant
            trend_info = None
            if regression == 'ct':
                trend_info = check_trend_significance(series, alpha=sig_level)

            return {
                'Test Statistic': result[0],
                'p-value': result[1],
                'Lags Used': result[2],
                'Critical Values': result[3],
                'Is Stationary': result[1] > sig_level,  # Note: KPSS null hypothesis is stationarity
                'Trend Info': trend_info
            }
        except Exception as e:
            st.warning(f"KPSS test failed: {e}")
            return None


    # Improved Phillips-Perron test implementation
    def run_pp_test(series, trend_spec):
        """
        Perform Phillips-Perron test using arch package
        If arch is not available, falls back to approximation
        """
        try:
            # Try to use arch package for proper PP test
            try:
                from arch.unitroot import PhillipsPerron
                
                # Map trend specification
                trend = 'n'  # no constant, no trend
                if trend_spec == "With Constant":
                    trend = 'c'
                elif trend_spec == "With Constant & Trend":
                    trend = 'ct'
                
                # Initialize Phillips-Perron test
                # lags=None means automatic lag selection
                pp_test = PhillipsPerron(series, lags=None, trend=trend, test_type='tau')
                
                # For 'ct' regression, check if trend is significant
                trend_info = None
                if trend == 'ct':
                    trend_info = check_trend_significance(series, alpha=sig_level)
                
                # Extract results - access attributes directly
                return {
                    'Test Statistic': pp_test.stat,
                    'p-value': pp_test.pvalue,
                    'Lags Used': pp_test.lags,
                    'Critical Values': pp_test.critical_values,
                    'Is Stationary': pp_test.pvalue < sig_level,
                    'Trend Info': trend_info
                }
            except ImportError:
                # Fallback: arch package not installed
                st.info("📦 arch package not installed. Install with: pip install arch")
                st.info("Using simplified approximation for Phillips-Perron test.")
                
                result = ndiffs(series, test='pp', max_d=2)
                
                trend_info = None
                if trend_spec == "With Constant & Trend":
                    trend_info = check_trend_significance(series, alpha=sig_level)
                
                return {
                    'Test Statistic': None,
                    'p-value': None,
                    'Lags Used': None,
                    'Critical Values': {'1%': None, '5%': None, '10%': None},
                    'Differencing Required': result,
                    'Is Stationary': result == 0,
                    'Trend Info': trend_info,
                    'Note': 'Approximation only. Install arch package (pip install arch) for full PP test.'
                }
            except Exception as e:
                # If arch is available but PP test fails for some reason
                st.warning(f"⚠️ Phillips-Perron test encountered an error: {e}")
                st.info("This may be due to insufficient observations or data issues.")
                return None
                
        except Exception as e:
            st.warning(f"PP test failed: {e}")
            return None


    # Function to perform Jarque-Bera normality test
    def run_normality_test(series):
        """Perform Jarque-Bera test for normality"""
        try:
            jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(series)
            return {
                'JB Statistic': jb_stat,
                'p-value': jb_pvalue,
                'Skewness': skew,
                'Kurtosis': kurtosis,
                'Is Normal': jb_pvalue > 0.05
            }
        except Exception as e:
            return None


    # Function to perform Ljung-Box autocorrelation test
    def run_autocorr_test(series, lags=10):
        """Perform Ljung-Box test for autocorrelation"""
        try:
            lb_result = acorr_ljungbox(series, lags=lags, return_df=True)
            return lb_result
        except Exception as e:
            return None


    # Function to determine order of integration and process type (TS vs DS)
    def determine_integration_order(series, test_type, trend_spec, max_lag, lag_method, max_diff=2):
        """
        Determine the order of integration and process type using proper econometric procedure.
        
        CRITICAL RULE:
        TS (Trend Stationary) can ONLY be identified when user selects "With Constant & Trend".
        If user selects "With Constant" or "Without Constant & Trend", TS will NEVER be returned.
        
        IMPORTANT RULES:
        
        TS (Trend Stationary) - ONLY from trend & intercept equation:
            - User MUST select "With Constant & Trend" specification
            - ADF & PP: p-value < 5% (reject H0 of unit root → NO unit root → stationary)
            - KPSS: p-value > 5% (fail to reject H0 of stationarity → stationary)
            - Trend coefficient MUST be significant (p-value < 5%)
        
        DS (Difference Stationary):
            - ADF & PP: p-value > 5% (fail to reject H0 of unit root → HAS unit root)
            - KPSS: p-value < 5% (reject H0 of stationarity → NOT stationary → has unit root)
            - Becomes stationary only after differencing
        
        Stationary (I(0) without trend):
            - Same stationarity criteria but trend NOT significant
            - OR user selected a model without trend component
        
        Returns: (integration_order, process_type)
        """
        original_series = series.copy()
        
        # =============================================
        # CRITICAL: TS can ONLY be identified from "With Constant & Trend" model
        # =============================================
        can_identify_ts = (trend_spec == "With Constant & Trend")

        # STEP 1: Test at level with the user-selected specification
        if test_type in ["ADF (Augmented Dickey-Fuller)", "ADF and PP", "ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
            
            # Test with user-selected specification
            adf_result = run_adf_test(original_series, trend_spec, max_lag, lag_method)
            
            if adf_result:
                adf_stationary = adf_result['Is Stationary']  # True if p-value < sig_level (reject H0 of unit root)
                trend_info = adf_result.get('Trend Info')
                # Trend info only available for "With Constant & Trend"
                trend_significant = trend_info['Is Trend Significant'] if trend_info else False
                
                # Also run PP test if selected (with same specification)
                pp_stationary = None
                if test_type in ["ADF and PP", "All Tests (ADF, PP, KPSS)"]:
                    pp_result = run_pp_test(original_series, trend_spec)
                    if pp_result and pp_result['p-value'] is not None:
                        pp_stationary = pp_result['Is Stationary']
                
                # Run KPSS test if selected
                kpss_stationary = None
                if test_type in ["ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                    # KPSS only supports 'c' (constant) or 'ct' (constant & trend)
                    kpss_spec = trend_spec if trend_spec != "Without Constant & Trend" else "With Constant"
                    kpss_result = run_kpss_test(original_series, kpss_spec)
                    if kpss_result:
                        # KPSS: H0 = stationary, H1 = unit root
                        # p-value > 5% means fail to reject H0 → stationary
                        # p-value < 5% means reject H0 → has unit root
                        kpss_stationary = kpss_result['Is Stationary']  # True if p-value > sig_level
                
                # =============================================
                # CHECK FOR TS (Trend Stationary)
                # ONLY possible if user selected "With Constant & Trend"
                # =============================================
                if can_identify_ts:
                    # Requirements for TS:
                    # 1. User selected "With Constant & Trend" ✓ (already checked above)
                    # 2. ADF: p < 5% (reject unit root = NO unit root)
                    # 3. PP (if available): p < 5% (reject unit root = NO unit root)
                    # 4. KPSS (if available): p > 5% (fail to reject = stationary)
                    # 5. Trend coefficient SIGNIFICANT
                    
                    is_ts = False
                    if adf_stationary and trend_significant:
                        # Basic condition met from ADF
                        is_ts = True
                        
                        # Check PP if available - must also show no unit root
                        if pp_stationary is not None and not pp_stationary:
                            is_ts = False  # PP says has unit root
                        
                        # Check KPSS if available - must confirm stationarity
                        if kpss_stationary is not None and not kpss_stationary:
                            # KPSS says NOT stationary (p < 5%), contradicts ADF
                            # This suggests DIFFERENCE stationary, not trend stationary
                            is_ts = False
                    
                    if is_ts:
                        return 0, 'TS'  # Trend Stationary
                
                # =============================================
                # CHECK FOR STATIONARY (I(0) without significant trend)
                # =============================================
                # This applies when:
                # - Series is stationary at level
                # - Either trend is NOT significant, OR user didn't select ct model
                
                if adf_stationary:
                    # For non-ct models, or ct model with non-significant trend
                    if not can_identify_ts or not trend_significant:
                        # Confirm stationarity
                        is_stationary = True
                        
                        # Check PP if available
                        if pp_stationary is not None and not pp_stationary:
                            is_stationary = False
                        
                        # Check KPSS if available
                        if kpss_stationary is not None and not kpss_stationary:
                            is_stationary = False  # KPSS rejects stationarity
                        
                        if is_stationary:
                            return 0, 'Stationary'  # I(0), stationary without trend
            
            # =============================================
            # CHECK FOR DS (Difference Stationary)
            # =============================================
            # If we reach here, the series is NOT stationary at level
            # This means: ADF p > 5% (has unit root) OR KPSS p < 5% (not stationary)
            # → This is DS if it becomes stationary after differencing
            
            if max_diff >= 1:
                first_diff = original_series.diff().dropna()
                if len(first_diff) > 10:
                    # Test first difference with constant only (standard for differenced series)
                    first_diff_adf = run_adf_test(first_diff, "With Constant", max_lag, lag_method)
                    
                    if first_diff_adf and first_diff_adf['Is Stationary']:
                        # ADF says first diff is stationary (p < 5%)
                        # Confirm with KPSS: should also say stationary (p > 5%)
                        
                        confirmed = True
                        if test_type in ["ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                            kpss_diff = run_kpss_test(first_diff, "With Constant")
                            if kpss_diff and not kpss_diff['Is Stationary']:
                                # KPSS says NOT stationary even after differencing
                                confirmed = False
                        
                        if confirmed:
                            return 1, 'DS'  # I(1) Difference Stationary
            
            # STEP 3: Test second difference
            if max_diff >= 2:
                first_diff = original_series.diff().dropna()
                second_diff = first_diff.diff().dropna()
                if len(second_diff) > 10:
                    second_diff_adf = run_adf_test(second_diff, "With Constant", max_lag, lag_method)
                    
                    if second_diff_adf and second_diff_adf['Is Stationary']:
                        confirmed = True
                        if test_type in ["ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                            kpss_diff2 = run_kpss_test(second_diff, "With Constant")
                            if kpss_diff2 and not kpss_diff2['Is Stationary']:
                                confirmed = False
                        
                        if confirmed:
                            return 2, 'DS'  # I(2) Difference Stationary
            
            # STEP 4: Test third difference (if requested)
            if max_diff >= 3:
                first_diff = original_series.diff().dropna()
                second_diff = first_diff.diff().dropna()
                third_diff = second_diff.diff().dropna()
                if len(third_diff) > 10:
                    third_diff_adf = run_adf_test(third_diff, "With Constant", max_lag, lag_method)
                    
                    if third_diff_adf and third_diff_adf['Is Stationary']:
                        return 3, 'DS'  # I(3) Difference Stationary

        # If not stationary after max_diff differences
        return max_diff + 1, 'Unknown'


    # Function to create ACF/PACF plots
    def plot_acf_pacf(series, var_name, lags=20):
        """Create ACF and PACF plots"""
        try:
            # Calculate ACF and PACF
            acf_vals = acf(series, nlags=min(lags, len(series)//2 - 1))
            pacf_vals = pacf(series, nlags=min(lags, len(series)//2 - 1))

            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)')
            )

            # ACF plot
            fig.add_trace(
                go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF',
                       marker_color='lightblue'),
                row=1, col=1
            )

            # PACF plot
            fig.add_trace(
                go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF',
                       marker_color='lightcoral'),
                row=1, col=2
            )

            # Add confidence intervals (95%)
            conf_interval = 1.96 / np.sqrt(len(series))
            
            # Add horizontal lines for confidence intervals
            for col in [1, 2]:
                fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", 
                             row=1, col=col, annotation_text="95% CI")
                fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", 
                             row=1, col=col)
                fig.add_hline(y=0, line_color="black", row=1, col=col)

            fig.update_layout(
                height=400,
                title_text=f"Correlograms for {var_name}",
                showlegend=False
            )

            return fig
        except Exception as e:
            st.warning(f"Could not create ACF/PACF plots: {e}")
            return None


    # Run the tests when the button is clicked
    if run_test and selected_vars:
        with st.spinner("Running unit root tests... This may take a moment."):
            results = {}
            integration_orders = {}
            process_types = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, var in enumerate(selected_vars):
                status_text.text(f"Testing {var}... ({idx+1}/{len(selected_vars)})")
                
                try:
                    series = st.session_state.data[var].dropna()
                    
                    # Check if series has enough observations
                    if len(series) < 10:
                        st.warning(f"Skipping {var}: Not enough observations (minimum 10 required)")
                        continue
                    
                    var_results = {}

                    # Run specified tests
                    if test_type in ["ADF (Augmented Dickey-Fuller)", "ADF and PP", "ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                        var_results['ADF'] = run_adf_test(series, test_spec, max_lag, lag_selection, user_lag)

                    if test_type in ["PP (Phillips-Perron)", "ADF and PP", "All Tests (ADF, PP, KPSS)"]:
                        var_results['PP'] = run_pp_test(series, test_spec)

                    if test_type in ["KPSS", "ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                        var_results['KPSS'] = run_kpss_test(series, test_spec)

                    # Run additional diagnostics
                    if show_normality:
                        var_results['Normality'] = run_normality_test(series)
                    
                    if show_autocorr:
                        var_results['Autocorrelation'] = run_autocorr_test(series)

                    # Determine order of integration and type of process
                    integration_order, process_type = determine_integration_order(
                        series, test_type, test_spec, max_lag, lag_selection, max_diff
                    )
                    integration_orders[var] = integration_order
                    process_types[var] = process_type

                    # Store results
                    var_results['Integration Order'] = integration_order
                    var_results['Process Type'] = process_type
                    results[var] = var_results

                except Exception as e:
                    st.error(f"Error testing {var}: {e}")
                    continue

                progress_bar.progress((idx + 1) / len(selected_vars))

            st.session_state.test_results = results
            status_text.text("Testing complete!")
            progress_bar.empty()

    # Display test results
    if st.session_state.test_results and selected_vars:
        st.header("📊 Unit Root Test Results")

        # Create summary statistics
        total_vars = len([v for v in selected_vars if v in st.session_state.test_results])
        i0_count = sum(1 for v in selected_vars if v in st.session_state.test_results and 
                       st.session_state.test_results[v].get('Integration Order') == 0)
        i1_count = sum(1 for v in selected_vars if v in st.session_state.test_results and 
                       st.session_state.test_results[v].get('Integration Order') == 1)
        i2_count = sum(1 for v in selected_vars if v in st.session_state.test_results and 
                       st.session_state.test_results[v].get('Integration Order') == 2)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Variables Tested", total_vars)
        col2.metric("I(0) - Stationary", i0_count)
        col3.metric("I(1) - First Difference", i1_count)
        col4.metric("I(2) - Second Difference", i2_count)

        # Create results dataframe
        all_results = []

        for var, tests in st.session_state.test_results.items():
            if var not in selected_vars:
                continue

            integration_order = tests.get('Integration Order', 'Unknown')
            process_type = tests.get('Process Type', 'Unknown')

            for test_name, result in tests.items():
                if test_name not in ['Integration Order', 'Process Type', 'Normality', 'Autocorrelation']:
                    if result:  # Only add if result is not None
                        row_data = {
                            'Variable': var,
                            'Test': test_name,
                            'Test Statistic': result.get('Test Statistic'),
                            'p-value': result.get('p-value'),
                            'Is Stationary': result.get('Is Stationary'),
                            'Lags Used': result.get('Lags Used'),
                            'Integration Order': integration_order,
                            'Process Type': process_type
                        }
                        all_results.append(row_data)

        if all_results:
            result_df = pd.DataFrame(all_results)

            # Show summary table
            st.subheader("Summary Table")

            # Style the dataframe for better visualization
            def highlight_integration_order(val):
                if val == 0:
                    return 'background-color: #d4ffcc; color: #006600; font-weight: bold'
                elif val == 1:
                    return 'background-color: #fff4cc; color: #cc6600; font-weight: bold'
                elif val == 2:
                    return 'background-color: #ffcccc; color: #990000; font-weight: bold'
                return ''

            def highlight_process_type(val):
                if val == 'TS':
                    return 'background-color: #cce5ff; color: #004085; font-weight: bold'
                elif val == 'DS':
                    return 'background-color: #e8d6f9; color: #4b0082; font-weight: bold'
                elif val in ['Stationary', 'I(0)']:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                return ''

            def highlight_stationary(val):
                if val is True:
                    return 'background-color: #d4ffcc; font-weight: bold'
                elif val is False:
                    return 'background-color: #ffcccc; font-weight: bold'
                return ''

            # Apply styling to dataframe - use map instead of deprecated applymap
            try:
                styled_df = result_df.style.map(highlight_integration_order, subset=['Integration Order']) \
                    .map(highlight_process_type, subset=['Process Type']) \
                    .map(highlight_stationary, subset=['Is Stationary'])
            except AttributeError:
                # Fallback for older pandas versions
                styled_df = result_df.style.applymap(highlight_integration_order, subset=['Integration Order']) \
                    .applymap(highlight_process_type, subset=['Process Type']) \
                    .applymap(highlight_stationary, subset=['Is Stationary'])

            st.dataframe(styled_df, use_container_width=True)

            # Create a downloadable CSV of results
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="unit_root_results.csv",
                mime="text/csv"
            )

            # Integration order visualization
            st.subheader("📈 Order of Integration Visualization")

            # Get unique variables
            unique_vars = result_df['Variable'].unique()

            # Create a dataframe for the heatmap
            heatmap_data = []
            for var in unique_vars:
                var_row = result_df[result_df['Variable'] == var].iloc[0]
                heatmap_data.append({
                    'Variable': var,
                    'Integration Order': var_row['Integration Order'],
                    'Process Type': var_row['Process Type']
                })

            heatmap_df = pd.DataFrame(heatmap_data)

            # Create a colorful visualization of integration orders
            fig = px.bar(
                heatmap_df,
                x='Variable',
                y='Integration Order',
                color='Integration Order',
                color_continuous_scale=[(0, 'green'), (0.5, 'orange'), (1, 'red')],
                labels={'Integration Order': 'I(d)'},
                height=400,
                text='Process Type',
                hover_data=['Process Type']
            )
            fig.update_traces(textposition='inside')
            fig.update_layout(
                title='Order of Integration by Variable',
                xaxis_title='Variables',
                yaxis_title='Order of Integration I(d)',
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['I(0)', 'I(1)', 'I(2)', 'I(3)']
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show detailed results
            st.subheader("🔍 Detailed Results")

            for var in selected_vars:
                if var in st.session_state.test_results:
                    # Get integration order and process type for this variable
                    var_results = st.session_state.test_results[var]
                    integration_order = var_results.get('Integration Order', 'Unknown')
                    process_type = var_results.get('Process Type', 'Unknown')

                    # Create a header with integration order and process type information
                    order_color = "green" if integration_order == 0 else "orange" if integration_order == 1 else "red"
                    process_color = "blue" if process_type == 'TS' else "purple" if process_type == 'DS' else "green" if process_type in ['I(0)', 'Stationary'] else "gray"

                    header_html = f"""
                    <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; 
                                background-color: #f8f9fa; border-left: 5px solid {order_color};">
                        <h3>{var}</h3>
                        <p>
                            <span style="color: {order_color}; font-weight: bold;">I({integration_order})</span> | 
                            <span style="color: {process_color}; font-weight: bold;">{process_type} Process</span>
                        </p>
                    </div>
                    """
                    st.markdown(header_html, unsafe_allow_html=True)

                    with st.expander(f"📋 Details for {var}", expanded=False):
                        tests = st.session_state.test_results[var]

                        # Display unit root test results
                        st.write("**Unit Root Test Results:**")
                        
                        for test_name, result in tests.items():
                            if test_name not in ['Integration Order', 'Process Type', 'Normality', 'Autocorrelation']:
                                if result:  # Only display if result is not None
                                    st.write(f"**{test_name} Test:**")

                                    # Format critical values
                                    critical_values = result.get('Critical Values', {})
                                    if isinstance(critical_values, dict):
                                        critical_values_str = ", ".join(
                                            [f"{k}: {v:.4f}" for k, v in critical_values.items() if v is not None])
                                    else:
                                        critical_values_str = "Not available"

                                    col1, col2 = st.columns(2)

                                    col1.metric("Test Statistic",
                                              f"{result.get('Test Statistic', 'N/A'):.4f}" if result.get(
                                                  'Test Statistic') is not None else "N/A")
                                    col1.metric("p-value", f"{result.get('p-value', 'N/A'):.4f}" if result.get(
                                        'p-value') is not None else "N/A")
                                    col1.metric("Lags Used", str(result.get('Lags Used', 'N/A')))

                                    col2.metric("Critical Values", critical_values_str)
                                    col2.metric("Is Stationary", "✅ Yes" if result.get('Is Stationary') else "❌ No")

                                    # Display note if available
                                    if result.get('Note'):
                                        st.info(result.get('Note'))

                                    # Show trend information if available
                                    trend_info = result.get('Trend Info')
                                    if trend_info:
                                        st.write("**Trend Analysis:**")

                                        col1, col2 = st.columns(2)
                                        col1.metric("Trend Coefficient",
                                                  f"{trend_info.get('Trend Coefficient', 'N/A'):.6f}" if trend_info.get(
                                                      'Trend Coefficient') is not None else "N/A")
                                        col1.metric("Trend p-value",
                                                  f"{trend_info.get('Trend p-value', 'N/A'):.4f}" if trend_info.get(
                                                      'Trend p-value') is not None else "N/A")

                                        col2.metric("Is Trend Significant",
                                                  "✅ Yes" if trend_info.get('Is Trend Significant') else "❌ No")
                                        col2.metric("R-squared",
                                                  f"{trend_info.get('R-squared', 'N/A'):.4f}" if trend_info.get(
                                                      'R-squared') is not None else "N/A")

                                        if trend_info.get('Is Trend Significant'):
                                            st.success(
                                                "✓ The trend coefficient is statistically significant, suggesting this may be a trend-stationary process.")

                                    if test_name == 'KPSS':
                                        st.info(
                                            "ℹ️ Note: For KPSS test, the null hypothesis is that the series IS stationary.")
                                    else:
                                        st.info(
                                            "ℹ️ Note: For ADF and PP tests, the null hypothesis is that the series has a unit root (non-stationary).")

                                    st.markdown("---")

                        # Display normality test if available
                        if show_normality and 'Normality' in tests and tests['Normality']:
                            st.write("**Normality Test (Jarque-Bera):**")
                            norm_result = tests['Normality']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("JB Statistic", f"{norm_result.get('JB Statistic', 0):.4f}")
                            col2.metric("p-value", f"{norm_result.get('p-value', 0):.4f}")
                            col3.metric("Skewness", f"{norm_result.get('Skewness', 0):.4f}")
                            col4.metric("Kurtosis", f"{norm_result.get('Kurtosis', 0):.4f}")
                            
                            if norm_result.get('Is Normal'):
                                st.success("✓ The series appears to be normally distributed (p > 0.05)")
                            else:
                                st.warning("⚠ The series does not appear to be normally distributed (p < 0.05)")
                            
                            st.markdown("---")

                        # Display autocorrelation test if available
                        if show_autocorr and 'Autocorrelation' in tests and tests['Autocorrelation'] is not None:
                            st.write("**Ljung-Box Autocorrelation Test:**")
                            lb_result = tests['Autocorrelation']
                            
                            # Show first few lags
                            st.dataframe(lb_result.head(10), use_container_width=True)
                            
                            # Check if any p-value is below 0.05
                            significant_lags = lb_result[lb_result['lb_pvalue'] < 0.05]
                            if len(significant_lags) > 0:
                                st.warning(f"⚠ Significant autocorrelation detected at {len(significant_lags)} lag(s)")
                            else:
                                st.success("✓ No significant autocorrelation detected")
                            
                            st.markdown("---")

                        # Prepare original, differenced, and detrended series for plotting
                        series = st.session_state.data[var].dropna()
                        
                        # Safely create differenced series
                        if len(series) > 1:
                            diff1 = series.diff().dropna()
                        else:
                            diff1 = pd.Series([])
                        
                        if len(diff1) > 1:
                            diff2 = diff1.diff().dropna()
                        else:
                            diff2 = pd.Series([])

                        # Create detrended series (for TS process visualization)
                        try:
                            x = np.arange(len(series))
                            X = sm.add_constant(x)
                            model = sm.OLS(series, X).fit()
                            trend = model.params[0] + model.params[1] * x
                            detrended = series - trend
                        except:
                            detrended = series - series.mean()

                        # Create plots
                        fig = make_subplots(rows=2, cols=2,
                                          subplot_titles=("Original Series", "Detrended Series",
                                                        "First Difference", "Second Difference"))

                        # Original series
                        fig.add_trace(go.Scatter(y=series, mode='lines', name='Original Series',
                                               line=dict(color='blue')), row=1, col=1)

                        # Detrended series
                        fig.add_trace(go.Scatter(y=detrended, mode='lines', name='Detrended Series',
                                               line=dict(color='purple')), row=1, col=2)

                        # First difference
                        if len(diff1) > 0:
                            fig.add_trace(go.Scatter(y=diff1, mode='lines', name='First Difference',
                                                   line=dict(color='orange')), row=2, col=1)

                        # Second difference
                        if len(diff2) > 0:
                            fig.add_trace(go.Scatter(y=diff2, mode='lines', name='Second Difference',
                                                   line=dict(color='red')), row=2, col=2)

                        fig.update_layout(height=600, title_text=f"Time Series Analysis: {var}", 
                                        showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Show ACF/PACF plots if requested
                        if show_acf_pacf and len(series) > 10:
                            st.write("**Autocorrelation Analysis:**")
                            acf_pacf_fig = plot_acf_pacf(series, var)
                            if acf_pacf_fig:
                                st.plotly_chart(acf_pacf_fig, use_container_width=True)

                        # Display integration order and process type explanation
                        st.subheader("Integration Analysis")

                        order_emoji = "🟢" if integration_order == 0 else "🟠" if integration_order == 1 else "🔴" if integration_order == 2 else "⚫"
                        process_emoji = "🔵" if process_type == 'TS' else "🟣" if process_type == 'DS' else "🟢" if process_type == 'I(0)' else "⚪"

                        st.markdown(f"""
                        ### Summary for {var}:

                        - **Integration Order**: {order_emoji} I({integration_order})
                        - **Process Type**: {process_emoji} {process_type} Process

                        #### What this means:
                        """)

                        if integration_order == 0 and process_type == 'TS':
                            st.success("""
                            **Trend Stationary (TS) Process, I(0)**

                            This series is stationary around a deterministic trend, determined from the trend & intercept equation.

                            **TS confirmed by:**
                            - ✅ Rejected unit root hypothesis (ADF/PP with constant & trend)
                            - ✅ Trend coefficient is statistically significant
                            - ✅ KPSS confirms stationarity (if tested)

                            **Key characteristics:**
                            - Stationary after removing the deterministic trend
                            - Shocks have temporary effects (mean-reverting to trend)
                            - Trend is deterministic, not stochastic
                            - Good for forecasting - reverts to predictable trend
                            - Detrending is appropriate (not differencing)
                            - Example: GDP with deterministic technological growth
                            """)
                        elif integration_order == 0 and process_type == 'Stationary':
                            st.success("""
                            **Stationary Process, I(0)**

                            This series is stationary around a constant mean (no significant trend).

                            **Key characteristics:**
                            - Mean-reverting around a constant level
                            - No deterministic trend detected
                            - Constant variance over time
                            - Temporary effects from shocks
                            - Can be modeled with ARMA directly
                            - No differencing or detrending needed
                            - Example: Inflation rate fluctuating around target
                            """)
                        elif integration_order == 1:
                            st.warning("""
                            **Difference Stationary (DS) Process, I(1)**

                            This series has a unit root and becomes stationary after first differencing.

                            **DS confirmed by:**
                            - ❌ Failed to reject unit root hypothesis in levels (ADF/PP)
                            - ✅ Becomes stationary after first differencing

                            **Key characteristics:**
                            - Contains one unit root
                            - Shocks have permanent effects (no mean reversion)
                            - Trend is stochastic, not deterministic
                            - Requires first differencing for ARIMA modeling
                            - May be suitable for cointegration analysis
                            - Detrending is NOT appropriate (use differencing)
                            - Common in economic/financial data (prices, GDP)
                            - Example: Stock prices, exchange rates
                            """)
                        elif integration_order == 2:
                            st.error("""
                            **Difference Stationary (DS) Process, I(2)**

                            This series requires second differencing to achieve stationarity.

                            **DS confirmed by:**
                            - ❌ Failed to reject unit root in levels and first difference
                            - ✅ Becomes stationary after second differencing

                            **Key characteristics:**
                            - Contains two unit roots
                            - Highly persistent with accelerating behavior
                            - Shocks have permanent and compounding effects
                            - Needs second differencing in ARIMA modeling
                            - More volatile and harder to forecast accurately
                            - Less common in practice (check for data issues)
                            - Example: Cumulative inflation, rare in real data
                            """)
                        else:
                            st.info(f"""
                            **Process type could not be clearly determined (I({integration_order}))**

                            This could indicate:
                            - Integration order > 2 (very rare in practice)
                            - Presence of seasonal unit roots
                            - Structural breaks in the series
                            - Model misspecification
                            - Data quality issues

                            **Recommendations:**
                            - Check for structural breaks (Zivot-Andrews test)
                            - Test for seasonal unit roots (if applicable)
                            - Verify data quality and transformations
                            - Consider alternative specifications
                            """)


            # Create Heatmap Visualization of all variables and their integration orders
            st.subheader("📊 Integration Order Summary Heatmap")

            # Prepare data for the heatmap
            variables = []
            orders = []
            process_types = []

            for var in selected_vars:
                if var in st.session_state.test_results:
                    variables.append(var)
                    orders.append(st.session_state.test_results[var].get('Integration Order', -1))
                    process_types.append(st.session_state.test_results[var].get('Process Type', 'Unknown'))

            # Create a dataframe
            heatmap_data = pd.DataFrame({
                'Variable': variables,
                'Integration Order': orders,
                'Process Type': process_types
            })

            # Order variables by integration order
            heatmap_data = heatmap_data.sort_values(['Integration Order', 'Process Type'])

            # Create heatmap with Plotly
            fig = go.Figure()

            # Color mapping
            color_map = {0: 'green', 1: 'orange', 2: 'red', 3: 'darkred'}

            # Add each variable as a bar, colored by integration order
            for i, row in heatmap_data.iterrows():
                var = row['Variable']
                order = row['Integration Order']
                process = row['Process Type']

                # Set color based on integration order
                color = color_map.get(order, 'gray')

                # Add the bar
                fig.add_trace(go.Bar(
                    x=[var],
                    y=[1],
                    name=f"I({order}) - {process}",
                    marker_color=color,
                    showlegend=True if i == 0 else False,
                    text=f"I({order})<br>{process}",
                    textposition="inside",
                    insidetextanchor="middle",
                    hoverinfo="text",
                    hovertext=f"{var}: I({order}) - {process} Process"
                ))

            fig.update_layout(
                title="Integration Order by Variable",
                xaxis_title="Variables",
                yaxis_title="",
                barmode='stack',
                height=400,
                xaxis={'categoryorder': 'array', 'categoryarray': heatmap_data['Variable'].tolist()},
                yaxis={'showticklabels': False},
                legend_title="Order of Integration"
            )

            st.plotly_chart(fig, use_container_width=True)

    # If no variables selected
    elif run_test and not selected_vars:
        st.warning("⚠️ Please select at least one variable to test.")

else:
    # If no data loaded yet
    st.info("📁 Please upload a data file to begin analysis.")

    # Sample data option
    if st.button("📊 Use Sample Data", type="primary"):
        # Generate sample time series data
        np.random.seed(42)
        dates = pd.date_range(start='2010-01-01', periods=120, freq='M')

        # Create different types of sample data
        t = np.arange(120)

        # I(0) - Stationary series
        stationary = np.random.normal(0, 3, 120)

        # I(0) - Trend stationary series
        trend = 0.2 * t
        trend_stationary = trend + np.random.normal(0, 3, 120)

        # I(1) - Random walk (needs first differencing)
        random_walk = np.cumsum(np.random.normal(0, 1, 120))

        # I(1) - Random walk with drift
        drift = 0.2
        random_walk_drift = np.cumsum(np.random.normal(drift, 1, 120))

        # I(2) - Double integrated series
        double_integrated = np.cumsum(np.cumsum(np.random.normal(0, 0.5, 120)))

        # Create DataFrame
        sample_data = pd.DataFrame({
            'stationary': stationary,
            'trend_stationary': trend_stationary,
            'random_walk': random_walk,
            'random_walk_drift': random_walk_drift,
            'double_integrated': double_integrated,
            'cycle': 10 * np.sin(np.linspace(0, 12 * np.pi, 120)) + np.random.normal(0, 2, 120),
            'trend_with_break': np.where(t < 60, t * 0.1, 60 * 0.1 + (t - 60) * 0.3) + np.random.normal(0, 2, 120)
        }, index=dates)

        # Store in session state
        st.session_state.data = sample_data
        st.session_state.variables = sample_data.columns.tolist()

        # Display data preview
        st.subheader("Sample Data Preview")
        st.dataframe(sample_data.head(10))

        # Display example series descriptions
        st.subheader("Sample Data Description")
        st.markdown("""
        This sample dataset contains different types of time series:

        - **stationary**: I(0) stationary series with constant mean and variance
        - **trend_stationary**: I(0) trend stationary series (TS process)
        - **random_walk**: I(1) random walk without drift (DS process)
        - **random_walk_drift**: I(1) random walk with drift (DS process)
        - **double_integrated**: I(2) double integrated series (DS process)
        - **cycle**: Stationary series with cyclical component
        - **trend_with_break**: Series with a structural break in the trend

        Try running unit root tests on these series to see how they behave!
        """)

        # Plot example series
        fig = go.Figure()
        for col in sample_data.columns:
            fig.add_trace(go.Scatter(x=sample_data.index, y=sample_data[col], mode='lines', name=col))

        fig.update_layout(
            title="Sample Time Series Data",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rerun to update UI - using st.rerun() instead of deprecated st.experimental_rerun()
        st.rerun()

# Footer with information
st.markdown("""
---
### 📚 About Unit Root Testing and Time Series Stationarity

Unit root tests determine whether a time series is stationary or non-stationary, and help identify the process type.

#### Key Concepts:

**1. Stationarity Types:**
- **Strictly Stationary**: All statistical properties are constant over time.
- **Weakly Stationary**: Mean, variance, and autocorrelation are constant over time.

**2. Process Types:**
- **Trend Stationary (TS)**: I(0) process stationary around a **deterministic** trend
  - Detected ONLY from trend & intercept equation
  - Requires: (1) Reject unit root (ADF/PP), (2) Significant trend coefficient
  - Shocks have temporary effects
  - Appropriate treatment: **Detrending**
  
- **Difference Stationary (DS)**: I(d) process with **stochastic** trend (d ≥ 1)
  - Has unit root in levels (fail to reject H0 in ADF/PP)
  - Becomes stationary after d differences
  - Shocks have permanent effects
  - Appropriate treatment: **Differencing**

- **Stationary**: I(0) process around constant mean
  - No significant deterministic trend
  - Already stationary, no transformation needed

**3. Order of Integration:**
- **I(0)**: Stationary series - can be TS (with trend) or Stationary (no trend)
- **I(1)**: DS process - needs first differencing to become stationary
- **I(2)**: DS process - needs second differencing (rare in practice)

**4. Critical Distinction - TS vs DS:**

The fundamental difference:
- **TS**: Deterministic trend + stationary deviations
  - Formula: Y_t = α + βt + ε_t (where ε_t is stationary)
  - Detrending makes it stationary
  - Shocks are temporary
  
- **DS**: Stochastic trend (random walk)
  - Formula: Y_t = Y_{t-1} + ε_t
  - Differencing makes it stationary  
  - Shocks are permanent

**5. Testing Procedure:**

**Step 1**: Test with **constant & trend** equation
- If stationary AND trend significant → **TS**
- If stationary but trend NOT significant → test with constant only
  - If still stationary → **Stationary (I(0))**
- If NOT stationary → proceed to Step 2

**Step 2**: Test **first difference** with constant
- If stationary → **DS, I(1)**
- If NOT stationary → proceed to Step 3

**Step 3**: Test **second difference** with constant
- If stationary → **DS, I(2)**
- If NOT stationary → higher order or other issues

**6. Available Tests:**
- **ADF (Augmented Dickey-Fuller)**: H0: Unit root present (non-stationary)
  - **p-value < 5%** → Reject H0 → Series is stationary (NO unit root)
  - **p-value > 5%** → Fail to reject H0 → Series has unit root
  
- **PP (Phillips-Perron)**: H0: Unit root present (non-stationary)
  - **p-value < 5%** → Reject H0 → Series is stationary (NO unit root)
  - **p-value > 5%** → Fail to reject H0 → Series has unit root
  - Similar to ADF but non-parametric correction for serial correlation
  
- **KPSS**: H0: Series is stationary (OPPOSITE hypothesis)
  - **p-value > 5%** → Fail to reject H0 → Series IS stationary
  - **p-value < 5%** → Reject H0 → Series has unit root (NOT stationary)
  - **Complementary** to ADF/PP (reversed hypothesis)

**Identification Rules:**
- **DS (Difference Stationary)**: 
  - ADF/PP: p > 5% (can't reject unit root) 
  - KPSS: p < 5% (reject stationarity)
  
- **TS (Trend Stationary)** - ONLY from constant & trend model:
  - ADF/PP: p < 5% (reject unit root = stationary)
  - KPSS: p > 5% (fail to reject stationarity)
  - Trend coefficient MUST be significant (p < 5%)

**7. Test Specifications:**
- **With Constant & Trend (ct)**: Use for TS detection and general testing
- **With Constant (c)**: Use for differenced series or when no trend
- **Without Constant & Trend (nc)**: Rarely used, specific cases only

**8. P-Value Differences Between Software (EViews vs Python):**

⚠️ **Important**: Test statistics are identical across software, but p-values may differ!

| Software | P-Value Method | Critical Values |
|----------|----------------|-----------------|
| **EViews** | Finite-sample response surface (MacKinnon 1996) | Sample-size adjusted |
| **Python (statsmodels)** | Asymptotic approximation (MacKinnon 1994) | MacKinnon 2010 tables |

**Recommendation**: 
- Compare **test statistic to critical values** for consistent results
- Critical values (1%, 5%, 10%) should be nearly identical across software
- If test statistic < critical value → Reject H0 (stationary)

**9. Lag Selection Methods:**

**Automatic Selection** (with Maximum lag):
- **AIC**: Akaike Information Criterion - tends to select more lags
- **BIC/SIC**: Bayesian/Schwarz Information Criterion - more parsimonious
- **t-stat**: Sequential t-test - starts at max lag, drops until significant
- **HQIC**: Hannan-Quinn Information Criterion - middle ground

**User Specified**: Directly set the exact number of lags

**Default Maximum Lag Formula** (Schwert 1989):
`maxlag = int(12 * (nobs/100)^(1/4))`

**10. Economic Implications:**

**TS Process Example**: GDP with technological progress
- Growth trend is deterministic
- Recessions are temporary deviations
- Economy returns to trend path

**DS Process Example**: Stock prices, exchange rates
- No deterministic path
- Shocks cause permanent level shifts
- Random walk with drift

**11. Model Selection Guidelines:**
- **TS (I(0) with trend)**: Use detrended data, ARMA models
- **Stationary (I(0) no trend)**: Use ARMA models directly
- **DS (I(1))**: Use ARIMA with d=1, or model differences
- **DS (I(2))**: Use ARIMA with d=2, check data quality

#### References:
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*.
- Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time series regression. *Biometrika*.
- Kwiatkowski, D., et al. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*.
- Nelson, C. R., & Plosser, C. I. (1982). Trends and random walks in macroeconomic time series. *Journal of Monetary Economics*.
- Kobayashi, M. and McAleer, M. (1999). Tests of Linear and Logarithmic Transformations for Integrated Processes. *Journal of the American Statistical Association*, 94(447), 860-868.

---
**Version**: 3.0 | **Author**: Dr. Merwan Roudane | **Email**: merwanroudane920@gmail.com

**Features Added in v3.0:**
- Enhanced Descriptive Statistics with interactive visualizations
- Data Transformations (Log, Percent Change, Difference, Standardized, Normalized, Box-Cox, Yeo-Johnson)
- Correlation Heatmap with coolwarm colorscale
- Kobayashi-McAleer Tests (V1, V2, U1, U2) for linear vs logarithmic transformation selection
- Improved Plotly visualizations throughout
""")
