# 📊 Portfolio Optimization & Analysis using Excel, Python & Streamlit

## 📌 Objective
To perform an in-depth analysis and optimization of a **diversified stock portfolio** using **real historical data**. This project simulates real-world portfolio management decisions by applying **Modern Portfolio Theory (MPT)** and visualizing key financial metrics such as **risk, return, correlation, efficient frontier, and asset allocation**.

---

## 🎯 Learning Outcomes
- Apply **Modern Portfolio Theory (MPT)** for portfolio construction.
- Compute and analyze **returns**, **volatility**, **covariance**, and **correlation** using Excel.
- Construct and **optimize a portfolio** using **Excel Solver**.
- Visualize insights and analytics using **Python** and **Streamlit**.
- Evaluate portfolio performance using **Sharpe Ratio** and **risk-return trade-off**.

---

## ⚙️ Tools & Technologies
- **Microsoft Excel** – Data processing, mathematical modeling, and Solver optimization.
- **Python (Pandas, Matplotlib, Seaborn, NumPy)** – Advanced visualizations and simulations.
- **Streamlit** – Interactive financial dashboard for scenario analysis.

---

## 📝 Project Workflow

### Step 1: 📥 Data Selection & Collection
- Selected 5–7 publicly listed Pakistani stocks from **diverse sectors** (Tech, Healthcare, Energy, Financials, etc.).
- Collected **daily adjusted closing prices** for the last 5 years from **Yahoo Finance/PSX**.
- Organized the dataset in Excel with each stock in a separate column, aligned by date.

### Step 2: 📈 Return Calculations
- Calculated daily returns:

=(Today's Price - Yesterday's Price) / Yesterday's Price


- Used Excel functions to compute:
- **Average Return**
- **Standard Deviation (Volatility)**
- **Covariance Matrix**
- **Correlation Matrix**

### Step 3: 📊 Data Visualization – Part 1
- Line charts for stock prices and returns.
- Correlation **heatmap** using Excel’s conditional formatting.
- **Histograms** of returns to observe volatility and distribution.

### Step 4: 🧮 Portfolio Metrics (Equal Weights)
- Assigned equal weights to all stocks.
- Calculated:
- **Expected Portfolio Return** using `=SUMPRODUCT(Weights, Mean Returns)`
- **Portfolio Variance** using matrix multiplication
- **Portfolio Standard Deviation** using `=SQRT(Variance)`

### Step 5: 🧠 Portfolio Optimization (Excel Solver)
- Maximized **Sharpe Ratio**:

(Portfolio Return - Risk-Free Rate) / Portfolio Std Dev


- Constraints:
- `SUM(Weights) = 1`
- Each Weight ≥ 0

- Obtained **optimized weights** from Solver.

### Step 6: 📊 Data Visualization – Part 2 (Python & Streamlit)
- **Pie chart** for optimized portfolio allocation.
- **Bar chart** for expected returns and risks.
- **Line graph** comparing cumulative returns (optimized vs. equal weights).
- **Efficient Frontier**:
- Simulated 1000+ random portfolios.
- Plotted return vs. risk (scatterplot).
- Highlighted optimal portfolio.

### Step 7: 📐 Performance Metrics
- **Sharpe Ratio** comparison for equal and optimized portfolios.
- Calculated **diversification benefit**.
- Analyzed **drawdowns** using cumulative returns.

### Step 8: 📲 Scenario Analysis (Streamlit Dashboard)
- Developed an interactive, macro-enabled dashboard.
- Switched between multiple views (correlation matrix, allocation chart, frontier).
- Included **scenario testing** (e.g., tech sector drop, interest rate change) for stress testing the portfolio.

---

## 📂 Folder Structure (Recommended)
portfolio-optimization/
│
├── excel file including data and models (final_fda_pro)
├── app.py/ # Python scripts and app interface
├── README.md # Project description and instructions
└── requirements.txt # Python dependencies


## 🚀 How to Run
1. Open the Excel file to view all calculations and Solver optimization.
2. Run the Streamlit dashboard:

streamlit run app.py

3. Use the dashboard to explore performance, visualizations, and scenario analysis.

---

## 📌 Conclusion
This project bridges **quantitative finance**, **data analysis**, and **visual storytelling**. By leveraging both Excel and Python, we created a comprehensive toolkit for portfolio construction, optimization, and decision-making—ideal for anyone looking to understand the fundamentals of **risk-managed investing**.
