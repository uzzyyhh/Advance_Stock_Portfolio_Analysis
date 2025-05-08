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
page = st.sidebar.radio("Select Page", ["Price Trends", "Returns", "Correlation", "Efficient Frontier", "Portfolio"])

# Sidebar for portfolio settings
st.sidebar.header("Portfolio Settings")
num_portfolios = st.sidebar.slider("Number of Portfolios to Simulate", 1000, 10000, 5000, step=1000)
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

# Calculate daily returns
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
    return mean_returns, cov_matrix

mean_returns, cov_matrix = calculate_stats(returns)

# Monte Carlo simulation for portfolio optimization
@st.cache_data
def monte_carlo_simulation(mean_returns, cov_matrix, num_portfolios):
    num_stocks = len(stocks)
    results = np.zeros((3 + num_stocks, num_portfolios))
    risk_free_rate = 0.1201 / 252  # 12.01% annualized risk-free rate, converted to daily

    for i in range(num_portfolios):
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe_ratio
        for j in range(num_stocks):
            results[3 + j, i] = weights[j]

    return results

results = monte_carlo_simulation(mean_returns, cov_matrix, num_portfolios)

# Find portfolios with max Sharpe ratio and min volatility
max_sharpe_idx = np.argmax(results[2])
min_vol_idx = np.argmin(results[1])

max_sharpe_portfolio = {
    'return': results[0, max_sharpe_idx] * 100,
    'risk': results[1, max_sharpe_idx] * 100,
    'sharpe': results[2, max_sharpe_idx],
    'weights': results[3:, max_sharpe_idx]
}

min_vol_portfolio = {
    'return': results[0, min_vol_idx] * 100,
    'risk': results[1, min_vol_idx] * 100,
    'sharpe': results[2, min_vol_idx],
    'weights': results[3:, min_vol_idx]
}

# Calculate cumulative returns
cumulative_returns = pd.DataFrame({
    'Date': returns['Date'],
    'Min Volatility': (returns[stocks].mul(min_vol_portfolio['weights'], axis=1).sum(axis=1) + 1).cumprod() - 1,
    'Max Sharpe': (returns[stocks].mul(max_sharpe_portfolio['weights'], axis=1).sum(axis=1) + 1).cumprod() - 1
})

# Calculate annualized returns for individual stocks
annualized_returns = mean_returns * 252 * 100  # Annualized return (%)

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
    
    # Option to select return type
    return_type = st.radio("Select Return Type", ["Daily Returns", "Annualized Returns", "Cumulative Returns"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{return_type} Over Time")
        if return_type == "Daily Returns":
            fig = px.line(returns, x='Date', y=selected_stock, title=f'{selected_stock} Daily Returns',
                          color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(xaxis_title="Date", yaxis_title="Daily Return", template="plotly_white")
        elif return_type == "Annualized Returns":
            annualized_df = pd.DataFrame({
                'Date': returns['Date'],
                selected_stock: returns[selected_stock] * 252  # Scale daily returns to annualized
            })
            fig = px.line(annualized_df, x='Date', y=selected_stock, title=f'{selected_stock} Annualized Returns (Scaled Daily)',
                          color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Return", template="plotly_white")
        else:  # Cumulative Returns
            cumulative_stock = (returns[selected_stock] + 1).cumprod() - 1
            cumulative_df = pd.DataFrame({
                'Date': returns['Date'],
                selected_stock: cumulative_stock * 100
            })
            fig = px.line(cumulative_df, x='Date', y=selected_stock, title=f'{selected_stock} Cumulative Returns',
                          color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return (%)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"{return_type} Distribution")
        if return_type == "Daily Returns":
            fig = px.histogram(returns, x=selected_stock, nbins=50, title=f'{selected_stock} Daily Returns Distribution',
                               color_discrete_sequence=['#FFD93D'])
            fig.update_layout(xaxis_title="Daily Return", yaxis_title="Count", template="plotly_white")
        elif return_type == "Annualized Returns":
            fig = px.histogram(returns, x=selected_stock, nbins=50, title=f'{selected_stock} Annualized Returns Distribution (Scaled Daily)',
                               color_discrete_sequence=['#FFD93D'])
            fig.update_layout(xaxis_title="Annualized Return", yaxis_title="Count", template="plotly_white")
        else:  # Cumulative Returns
            cumulative_stock = (returns[selected_stock] + 1).cumprod() - 1
            cumulative_df = pd.DataFrame({selected_stock: cumulative_stock * 100})
            fig = px.histogram(cumulative_df, x=selected_stock, nbins=50, title=f'{selected_stock} Cumulative Returns Distribution',
                               color_discrete_sequence=['#FFD93D'])
            fig.update_layout(xaxis_title="Cumulative Return (%)", yaxis_title="Count", template="plotly_white")
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

# Page: Efficient Frontier
if page == "Efficient Frontier":
    st.header("Efficient Frontier")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results[1] * 100,
        y=results[0] * 100,
        mode='markers',
        marker=dict(
            size=5,
            color=results[2],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name='Portfolios'
    ))
    fig.add_trace(go.Scatter(
        x=[min_vol_portfolio['risk']],
        y=[min_vol_portfolio['return']],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name='Min Volatility'
    ))
    fig.add_trace(go.Scatter(
        x=[max_sharpe_portfolio['risk']],
        y=[max_sharpe_portfolio['return']],
        mode='markers',
        marker=dict(size=12, color='green', symbol='star'),
        name='Max Sharpe'
    ))
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Risk (%)",
        yaxis_title="Return (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# Page: Portfolio
if page == "Portfolio":
    st.header("Portfolio Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weights (Min Volatility Portfolio)")
        weights_df = pd.DataFrame({
            'Stock': stocks,
            'Weight': min_vol_portfolio['weights'] * 100
        })
        fig = px.pie(weights_df, values='Weight', names='Stock', title='Min Volatility Portfolio Weights',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD93D', '#1A936F', '#C06C84'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Weights (Max Sharpe Portfolio)")
        weights_df = pd.DataFrame({
            'Stock': stocks,
            'Weight': max_sharpe_portfolio['weights'] * 100
        })
        fig = px.pie(weights_df, values='Weight', names='Stock', title='Max Sharpe Portfolio Weights',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD93D', '#1A936F', '#C06C84'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Metrics")
    stats_df = pd.DataFrame({
        'Portfolio': ['Min Volatility', 'Max Sharpe'],
        'Annual Return (%)': [min_vol_portfolio['return'], max_sharpe_portfolio['return']],
        'Annual Risk (%)': [min_vol_portfolio['risk'], max_sharpe_portfolio['risk']],
        'Sharpe Ratio': [min_vol_portfolio['sharpe'], max_sharpe_portfolio['sharpe']]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stats_df['Portfolio'], y=stats_df['Annual Return (%)'], name='Annual Return (%)',
                         marker_color='#FF6B6B'))
    fig.add_trace(go.Bar(x=stats_df['Portfolio'], y=stats_df['Annual Risk (%)'], name='Annual Risk (%)',
                         marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(x=stats_df['Portfolio'], y=stats_df['Sharpe Ratio'], name='Sharpe Ratio',
                         marker_color='#FFD93D'))
    fig.update_layout(barmode='group', title="Performance Comparison", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns['Date'], y=cumulative_returns['Min Volatility'] * 100,
                             mode='lines', name='Min Volatility', line=dict(color='#FFD93D')))
    fig.add_trace(go.Scatter(x=cumulative_returns['Date'], y=cumulative_returns['Max Sharpe'] * 100,
                             mode='lines', name='Max Sharpe', line=dict(color='#1A936F')))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
