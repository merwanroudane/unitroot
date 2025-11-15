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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Unit Root Testing App",
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
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🔍 Advanced Time Series Unit Root Testing Dashboard")
st.markdown("""
This application provides comprehensive unit root testing capabilities for time series data.
Upload your data file, select variables to test, and configure test parameters to analyze stationarity.
The app distinguishes between **Trend Stationary (TS)** and **Difference Stationary (DS)** processes,
and determines the order of integration: I(0), I(1), or I(2).
""")

# Sidebar for inputs
st.sidebar.header("Configuration")

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

# Handle uploaded file
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Store in session state
        st.session_state.data = data
        st.session_state.variables = data.columns.tolist()

        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10))

        # Basic data info
        st.subheader("Data Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{data.shape[0]}")
        col2.metric("Columns", f"{data.shape[1]}")
        col3.metric("Missing Values", f"{data.isna().sum().sum()}")
        col4.metric("Data Type", "Numeric" if data.select_dtypes(include=[np.number]).shape[1] > 0 else "Mixed")

        # Display descriptive statistics
        with st.expander("📊 Descriptive Statistics"):
            st.dataframe(data.describe())

    except Exception as e:
        st.error(f"Error loading file: {e}")

# If data is loaded, show options
if st.session_state.data is not None:
    # Variable selection
    st.sidebar.subheader("Variable Selection")
    all_vars = st.sidebar.checkbox("Select All Variables", key="all_vars")

    if all_vars:
        selected_vars = st.session_state.variables
    else:
        selected_vars = st.sidebar.multiselect(
            "Select Variables for Testing",
            options=st.session_state.variables,
            default=st.session_state.selected_vars
        )

    st.session_state.selected_vars = selected_vars

    # Test configuration
    st.sidebar.subheader("Test Configuration")

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

    # Lag selection
    lag_selection = st.sidebar.selectbox(
        "Lag Selection Method",
        options=["AIC (Akaike Information Criterion)",
                 "BIC (Bayesian Information Criterion)",
                 "t-stat (Sequential t-test)",
                 "HQIC (Hannan-Quinn Information Criterion)",
                 "User Specified"],
        index=0
    )

    # Show user lag input if user specified
    max_lag = 12
    if lag_selection == "User Specified":
        max_lag = st.sidebar.number_input("Maximum Lag Length", min_value=1, max_value=50, value=12)

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
    def run_adf_test(series, trend_spec, max_lag, lag_method):
        """Perform Augmented Dickey-Fuller test"""
        try:
            trend = 'nc'  # no constant, no trend
            if trend_spec == "With Constant":
                trend = 'c'
            elif trend_spec == "With Constant & Trend":
                trend = 'ct'

            # Map lag selection method to statsmodels parameter
            ic_map = {
                "AIC (Akaike Information Criterion)": "aic",
                "BIC (Bayesian Information Criterion)": "bic",
                "t-stat (Sequential t-test)": "t-stat",
                "HQIC (Hannan-Quinn Information Criterion)": "hqic",
                "User Specified": None
            }

            ic = ic_map[lag_method]

            if ic:
                result = adfuller(series, maxlag=max_lag, regression=trend, autolag=ic)
            else:
                result = adfuller(series, maxlag=max_lag, regression=trend, autolag=None)

            # For 'ct' regression, check if trend is significant
            trend_info = None
            if trend == 'ct':
                trend_info = check_trend_significance(series, alpha=sig_level)

            return {
                'Test Statistic': result[0],
                'p-value': result[1],
                'Lags Used': result[2],
                'Observations Used': result[3],
                'Critical Values': result[4],
                'Is Stationary': result[1] < sig_level,
                'Trend Info': trend_info
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
        
        TS (Trend Stationary): 
            - ONLY concluded from trend & intercept equation
            - Requires: (1) Reject H0 of unit root (ADF/PP), (2) Trend coefficient significant
            - Optional: Confirm with KPSS (accept H0 of stationarity)
        
        DS (Difference Stationary): 
            - Has unit root in levels (fail to reject H0 in ADF/PP)
            - Becomes stationary after d differences
        
        Stationary: 
            - I(0) without deterministic trend
        
        Returns: (integration_order, process_type)
        """
        original_series = series.copy()

        # STEP 1: Test with TREND & INTERCEPT equation (required for TS detection)
        if test_type in ["ADF (Augmented Dickey-Fuller)", "ADF and PP", "ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
            
            # Test with constant and trend
            adf_ct = run_adf_test(original_series, "With Constant & Trend", max_lag, lag_method)
            
            # Check for TS: must be stationary AND trend must be significant
            if adf_ct and adf_ct['Is Stationary']:
                trend_info = adf_ct.get('Trend Info')
                
                # CONDITION FOR TS: Stationary from trend & intercept AND trend significant
                if trend_info and trend_info['Is Trend Significant']:
                    
                    # Optional: Confirm with KPSS if available
                    if test_type in ["ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                        kpss_ct = run_kpss_test(original_series, "With Constant & Trend")
                        
                        if kpss_ct and kpss_ct['Is Stationary']:
                            # Both ADF and KPSS confirm stationarity, trend significant → TS
                            return 0, 'TS'
                        else:
                            # ADF says stationary but KPSS says not - inconclusive
                            # Since trend is significant and ADF rejects unit root, lean towards TS
                            return 0, 'TS'
                    else:
                        # Only ADF available, trend significant → TS
                        return 0, 'TS'
                
                else:
                    # Stationary but trend NOT significant
                    # Test with just constant to confirm I(0) stationary
                    adf_c = run_adf_test(original_series, "With Constant", max_lag, lag_method)
                    
                    if adf_c and adf_c['Is Stationary']:
                        # Confirm with KPSS if available
                        if test_type in ["ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                            kpss_c = run_kpss_test(original_series, "With Constant")
                            if kpss_c and kpss_c['Is Stationary']:
                                return 0, 'Stationary'  # I(0), no trend
                        return 0, 'Stationary'  # I(0), no trend
            
            # STEP 2: NOT stationary at level → Test for DS by differencing
            # Has unit root (failed to reject H0 in ADF with trend & intercept)
            
            if max_diff >= 1:
                first_diff = original_series.diff().dropna()
                if len(first_diff) > 10:
                    # Test first difference with constant (standard for differenced series)
                    first_diff_result = run_adf_test(first_diff, "With Constant", max_lag, lag_method)
                    
                    if first_diff_result and first_diff_result['Is Stationary']:
                        # Confirm with KPSS if available
                        if test_type in ["ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                            kpss_diff = run_kpss_test(first_diff, "With Constant")
                            if kpss_diff and kpss_diff['Is Stationary']:
                                return 1, 'DS'  # I(1) Difference Stationary
                        
                        # Unit root in levels, stationary after 1st diff → DS
                        return 1, 'DS'
            
            # STEP 3: Test second difference
            if max_diff >= 2:
                second_diff = first_diff.diff().dropna()
                if len(second_diff) > 10:
                    second_diff_result = run_adf_test(second_diff, "With Constant", max_lag, lag_method)
                    
                    if second_diff_result and second_diff_result['Is Stationary']:
                        # Confirm with KPSS if available
                        if test_type in ["ADF and KPSS", "All Tests (ADF, PP, KPSS)"]:
                            kpss_diff2 = run_kpss_test(second_diff, "With Constant")
                            if kpss_diff2 and kpss_diff2['Is Stationary']:
                                return 2, 'DS'  # I(2) Difference Stationary
                        
                        return 2, 'DS'  # I(2) Difference Stationary
            
            # STEP 4: Test third difference (if requested)
            if max_diff >= 3:
                third_diff = second_diff.diff().dropna()
                if len(third_diff) > 10:
                    third_diff_result = run_adf_test(third_diff, "With Constant", max_lag, lag_method)
                    
                    if third_diff_result and third_diff_result['Is Stationary']:
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
                        var_results['ADF'] = run_adf_test(series, test_spec, max_lag, lag_selection)

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
  - Reject H0 → Series is stationary
  - Fail to reject H0 → Series has unit root
  
- **PP (Phillips-Perron)**: H0: Unit root present (non-stationary)
  - Similar to ADF but non-parametric correction for serial correlation
  - More robust to heteroskedasticity
  
- **KPSS**: H0: Series is stationary
  - Reject H0 → Series is non-stationary
  - Fail to reject H0 → Series is stationary
  - **Complementary** to ADF/PP (reversed hypothesis)

**7. Test Specifications:**
- **With Constant & Trend (ct)**: Use for TS detection and general testing
- **With Constant (c)**: Use for differenced series or when no trend
- **Without Constant & Trend (nc)**: Rarely used, specific cases only

**8. Lag Selection Methods:**
- **AIC**: Akaike Information Criterion (allows more lags)
- **BIC**: Bayesian Information Criterion (more parsimonious)
- **t-stat**: Sequential t-test for lag significance
- **HQIC**: Hannan-Quinn Information Criterion (middle ground)

**9. Economic Implications:**

**TS Process Example**: GDP with technological progress
- Growth trend is deterministic
- Recessions are temporary deviations
- Economy returns to trend path

**DS Process Example**: Stock prices, exchange rates
- No deterministic path
- Shocks cause permanent level shifts
- Random walk with drift

**10. Model Selection Guidelines:**
- **TS (I(0) with trend)**: Use detrended data, ARMA models
- **Stationary (I(0) no trend)**: Use ARMA models directly
- **DS (I(1))**: Use ARIMA with d=1, or model differences
- **DS (I(2))**: Use ARIMA with d=2, check data quality

#### References:
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*.
- Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time series regression. *Biometrika*.
- Kwiatkowski, D., et al. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*.
- Nelson, C. R., & Plosser, C. I. (1982). Trends and random walks in macroeconomic time series. *Journal of Monetary Economics*.

---
**Version**: 2.1 | **Developed for**: Advanced Econometric Time Series Analysis
""")
