import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Streamlit page configuration
st.set_page_config(page_title="Stock Portfolio Analysis", layout="wide")

# Title
st.title("Stock Portfolio Analysis")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Price Trends", "Returns", "Correlation", "Portfolio"])

# Sidebar for portfolio simulation
st.sidebar.header("Portfolio Settings")
num_portfolios = st.sidebar.slider("Number of Simulated Portfolios", 500, 5000, 1000, step=500)

st.sidebar.markdown(
    f"""
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

# Calculate returns
@st.cache_data
def calculate_returns(data):
    returns = data[stocks].pct_change().dropna()
    returns['Date'] = data['Date'][1:].values
    return returns

returns = calculate_returns(data)

# Calculate statistics
@st.cache_data
def calculate_stats(returns):
    mean_returns = returns[stocks].mean()
    cov_matrix = returns[stocks].cov()
    std_devs = np.sqrt(np.diag(cov_matrix))
    return mean_returns, cov_matrix, std_devs

mean_returns, cov_matrix, std_devs = calculate_stats(returns)

# Portfolio optimization
@st.cache_data
def optimize_portfolio(mean_returns, cov_matrix, num_portfolios):
    num_stocks = len(stocks)
    portfolios = []
    for _ in range(num_portfolios):
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_risk = np.sqrt(np.sum(np.dot(weights, np.dot(cov_matrix, weights.T))))
        portfolios.append({'weights': weights, 'return': portfolio_return, 'risk': portfolio_risk})
    
    portfolios_df = pd.DataFrame(portfolios)
    risk_free_rate = 0.02 / 252
    portfolios_df['sharpe'] = (portfolios_df['return'] - risk_free_rate) / portfolios_df['risk']
    optimal_portfolio = portfolios_df.loc[portfolios_df['sharpe'].idxmax()]
    
    equal_weights = np.array([1/num_stocks] * num_stocks)
    equal_return = np.sum(mean_returns * equal_weights)
    equal_risk = np.sqrt(np.sum(np.dot(equal_weights, np.dot(cov_matrix, equal_weights.T))))
    
    return portfolios_df, optimal_portfolio, equal_weights, equal_return, equal_risk

portfolios_df, optimal_portfolio, equal_weights, equal_return, equal_risk = optimize_portfolio(mean_returns, cov_matrix, num_portfolios)

# Calculate cumulative returns
cumulative_returns = pd.DataFrame({
    'Date': returns['Date'],
    'Equal': (returns[stocks].mean(axis=1) + 1).cumprod() - 1,
    'Optimal': (returns[stocks].mul(optimal_portfolio['weights'], axis=1).sum(axis=1) + 1).cumprod() - 1
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
        st.subheader("Portfolio Weights")
        weights_df = pd.DataFrame({
            'Stock': stocks,
            'Weight': optimal_portfolio['weights'] * 100
        })
        fig = px.pie(weights_df, values='Weight', names='Stock', title='Optimal Portfolio Weights',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD93D', '#1A936F', '#C06C84'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Expected Returns & Risk")
        stats_df = pd.DataFrame({
            'Stock': stocks,
            'Return': mean_returns * 100,
            'Risk': std_devs * 100
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stats_df['Stock'], y=stats_df['Return'], name='Expected Return (%)',
                             marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(x=stats_df['Stock'], y=stats_df['Risk'], name='Risk (Std Dev %)',
                             marker_color='#4ECDC4'))
        fig.update_layout(barmode='group', title="Expected Returns & Risk", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Cumulative Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns['Date'], y=cumulative_returns['Equal'] * 100,
                             mode='lines', name='Equal-Weight', line=dict(color='#FFD93D')))
    fig.add_trace(go.Scatter(x=cumulative_returns['Date'], y=cumulative_returns['Optimal'] * 100,
                             mode='lines', name='Optimized', line=dict(color='#1A936F')))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Efficient Frontier")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolios_df['risk'] * 100,
        y=portfolios_df['return'] * 100,
        mode='markers',
        name='Portfolios',
        marker=dict(size=5, color='#C06C84')
    ))
    fig.add_trace(go.Scatter(
        x=[optimal_portfolio['risk'] * 100],
        y=[optimal_portfolio['return'] * 100],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(size=12, color='#F28C38', symbol='star')
    ))
    fig.add_trace(go.Scatter(
        x=[equal_risk * 100],
        y=[equal_return * 100],
        mode='markers',
        name='Equal-Weight Portfolio',
        marker=dict(size=12, color='#6C5B7B', symbol='triangle-up')
    ))
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Risk (Std Dev %)",
        yaxis_title="Expected Return (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
