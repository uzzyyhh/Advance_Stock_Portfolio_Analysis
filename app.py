import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# Streamlit page configuration
st.set_page_config(page_title="Stock Portfolio Analysis", layout="wide")

# Title
st.title("Stock Portfolio Analysis")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Price Trends", "Returns", "Correlation", "Portfolio"])

# Sidebar for portfolio settings
st.sidebar.header("Portfolio Settings")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 5.0, 15.0, 12.01, step=0.01)
st.sidebar.markdown(
    """
    **Optimization Method**: Mean-variance optimization maximizing the Sharpe Ratio, with constraints for full allocation and non-negative weights (no short-selling).
    """
)
st.sidebar.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExeDFkZTBuNDN5OTdjdTQ5dXdtYW95cHljdmc0bDU5dTV6cWFvYjN0bSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4qbCWW4WwDpoYtzj6S/giphy.gif" style="width:100%; max-width:100%; height:auto;">
    </div>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel("final_fda_pro.xlsx")
    return data.dropna()

data = load_data()
stocks = ['CNERGY.KA', 'SNGP.KA', 'PAEL.KA', 'POWER.KA', 'WTL.KA']
sectors = ['Energy', 'Utilities', 'Industrials', 'Construction Materials', 'Telecommunications']

# Calculate daily returns
@st.cache_data
def calculate_returns(data):
    returns = data[stocks].pct_change().dropna()
    returns['Date'] = data['Date'][1:].values
    return returns

returns = calculate_returns(data)

# Calculate statistics (annualized)
@st.cache_data
def calculate_stats(returns):
    trading_days = 252
    mean_returns = returns[stocks].mean() * trading_days * 100  # Annualized return (%)
    cov_matrix = returns[stocks].cov() * trading_days * 100  # Annualized covariance
    std_devs = np.sqrt(np.diag(cov_matrix))  # Annualized standard deviation
    return mean_returns, cov_matrix, std_devs

mean_returns, cov_matrix, std_devs = calculate_stats(returns)

# Portfolio optimization using mean-variance optimization
@st.cache_data
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_stocks = len(stocks)

    # Objective function: Negative Sharpe Ratio (to maximize)
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -((portfolio_return - risk_free_rate) / portfolio_risk)

    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds: Non-negative weights (no short-selling)
    bounds = tuple((0, 1) for _ in range(num_stocks))
    # Initial guess: Equal weights
    init_weights = np.array([1/num_stocks] * num_stocks)

    # Optimize
    result = minimize(
        neg_sharpe_ratio,
        init_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    optimal_return = np.sum(mean_returns * optimal_weights)
    optimal_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_risk

    # Equal-weight portfolio (initial portfolio)
    equal_weights = np.array([1/num_stocks] * num_stocks)
    equal_return = np.sum(mean_returns * equal_weights)
    equal_risk = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
    equal_sharpe = (equal_return - risk_free_rate) / equal_risk

    return {
        'optimal': {'weights': optimal_weights, 'return': optimal_return, 'risk': optimal_risk, 'sharpe': optimal_sharpe},
        'equal': {'weights': equal_weights, 'return': equal_return, 'risk': equal_risk, 'sharpe': equal_sharpe}
    }

portfolio_data = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)

# Calculate cumulative returns (scaled to annualized perspective)
cumulative_returns = pd.DataFrame({
    'Date': returns['Date'],
    'Equal': (returns[stocks].mean(axis=1) + 1).cumprod() * 100,
    'Optimal': (returns[stocks].mul(portfolio_data['optimal']['weights'], axis=1).sum(axis=1) + 1).cumprod() * 100
})

# Page: Price Trends
if page == "Price Trends":
    st.header("Stock Price Trends")
    selected_stock = st.selectbox("Select Stock", stocks)
    fig = px.line(data, x='Date', y=selected_stock, title=f'{selected_stock} Price Trend',
                  color_discrete_sequence=['#FF6B6B'])
    fig.update_layout(xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Page: Returns
if page == "Returns":
    st.header("Stock Returns")
    selected_stock = st.selectbox("Select Stock", stocks)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Returns Over Time")
        fig = px.line(returns, x='Date', y=selected_stock, title=f'{selected_stock} Daily Returns',
                      color_discrete_sequence=['#4ECDC4'])
        fig.update_layout(xaxis_title="Date", yaxis_title="Daily Return", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Returns Distribution")
        fig = px.histogram(returns, x=selected_stock, nbins=50, title=f'{selected_stock} Returns Distribution',
                           color_discrete_sequence=['#FFD93D'])
        fig.update_layout(xaxis_title="Daily Return", yaxis_title="Count", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# Page: Correlation
if page == "Correlation":
    st.header("Correlation Matrix")
    corr_matrix = returns[stocks].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=stocks,
        y=stocks,
        colorscale='Plasma',
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"}
    ))
    fig.update_layout(title="Correlation Matrix", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Page: Portfolio
if page == "Portfolio":
    st.header("Portfolio Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sector Allocation")
        weights_df = pd.DataFrame({
            'Stock': stocks,
            'Sector': sectors,
            'Weight': portfolio_data['optimal']['weights'] * 100
        })
        fig = px.pie(weights_df, values='Weight', names='Sector', title='Optimized Portfolio Sector Allocation',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD93D', '#1A936F', '#C06C84'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Return and Risk Comparison")
        stats_df = pd.DataFrame({
            'Portfolio': ['Initial (Equal)', 'Optimized'],
            'Return (%)': [portfolio_data['equal']['return'], portfolio_data['optimal']['return']],
            'Risk (%)': [portfolio_data['equal']['risk'], portfolio_data['optimal']['risk']]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stats_df['Portfolio'], y=stats_df['Return (%)'], name='Return (%)',
                             marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(x=stats_df['Portfolio'], y=stats_df['Risk (%)'], name='Risk (%)',
                             marker_color='#4ECDC4'))
        fig.update_layout(barmode='group', title="Return and Risk Comparison", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sharpe Ratio Comparison")
        sharpe_df = pd.DataFrame({
            'Portfolio': ['Initial (Equal)', 'Optimized'],
            'Sharpe Ratio': [portfolio_data['equal']['sharpe'], portfolio_data['optimal']['sharpe']]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sharpe_df['Portfolio'], y=sharpe_df['Sharpe Ratio'], name='Sharpe Ratio',
                             marker_color='#FFD93D'))
        fig.update_layout(title="Sharpe Ratio Comparison", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk vs. Return")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[portfolio_data['equal']['risk']], y=[portfolio_data['equal']['return']],
            mode='markers+text', name='Initial (Equal)', marker=dict(size=15, color='#C06C84', symbol='circle'),
            text=['Initial'], textposition='top center'
        ))
        fig.add_trace(go.Scatter(
            x=[portfolio_data['optimal']['risk']], y=[portfolio_data['optimal']['return']],
            mode='markers+text', name='Optimized', marker=dict(size=15, color='#F28C38', symbol='star'),
            text=['Optimized'], textposition='top center'
        ))
        fig.add_hline(y=risk_free_rate, line_dash="dash", line_color="black",
                      annotation_text=f"T-Bill ({risk_free_rate}%)", annotation_position="bottom right")
        fig.update_layout(
            title="Risk vs. Return",
            xaxis_title="Annualized Risk (%)",
            yaxis_title="Annualized Return (%)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns['Date'], y=cumulative_returns['Equal'],
                             mode='lines', name='Initial (Equal)', line=dict(color='#FFD93D')))
    fig.add_trace(go.Scatter(x=cumulative_returns['Date'], y=cumulative_returns['Optimal'],
                             mode='lines', name='Optimized', line=dict(color='#1A936F')))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Growth (%)",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
