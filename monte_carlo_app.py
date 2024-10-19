import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit App Configuration
st.title("Monte Carlo Portfolio Simulation")
st.write("An interactive tool to simulate portfolio growth using Monte Carlo methods. Adjust the parameters below:")

# User Inputs for the Model
initial_investment = st.number_input('Initial Investment ($)', min_value=0, value=10000, step=1000)
years = st.slider('Number of Years', min_value=1, max_value=50, value=20)
simulations = st.slider('Number of Simulations', min_value=100, max_value=10000, value=1000, step=100)
average_return = st.number_input('Average Return (e.g., 0.08 for 8%)', min_value=0.0, max_value=1.0, value=0.08, step=0.01)
std_dev = st.number_input('Standard Deviation (Volatility)', min_value=0.0, max_value=1.0, value=0.15, step=0.01)
annual_contribution = st.number_input('Annual Contribution ($)', min_value=0, value=5000, step=500)
discount_rate = st.number_input('Discount Rate for NPV (e.g., 0.05 for 5%)', min_value=0.0, max_value=1.0, value=0.05, step=0.01)
random_seed = st.number_input('Random Seed (for reproducibility)', min_value=0, value=42, step=1)

# Set the random seed for reproducibility
np.random.seed(random_seed)

# Run Monte Carlo Simulation
run_simulation = st.button("Run Simulation")

if run_simulation:
    # Initialize simulation arrays
    portfolio_values = np.zeros((simulations, years + 1))
    portfolio_values[:, 0] = initial_investment
    npv_over_time = np.zeros((simulations, years + 1))

    # Monte Carlo simulations
    for i in range(simulations):
        for year in range(1, years + 1):
            if year == 1:
                random_return = np.random.normal(average_return, std_dev)
            else:
                # Correlate current year's movement with the previous year
                if portfolio_values[i, year - 1] > portfolio_values[i, year - 2]:
                    # Previous year was up
                    random_return = np.random.normal(average_return, std_dev) if np.random.rand() < 0.7 else np.random.normal(-average_return, std_dev)
                else:
                    # Previous year was down
                    random_return = np.random.normal(-average_return, std_dev) if np.random.rand() < 0.7 else np.random.normal(average_return, std_dev)

            # Update portfolio value with annual contribution and return
            portfolio_values[i, year] = (portfolio_values[i, year - 1] + annual_contribution) * (1 + random_return)

            # Calculate NPV over time
            if year == 1:
                npv_over_time[i, year] = initial_investment / ((1 + discount_rate) ** 0)  # Discount initial investment (which is itself)
            else:
                npv_over_time[i, year] = npv_over_time[i, year - 1] + (annual_contribution / ((1 + discount_rate) ** year)) + \
                                         ((portfolio_values[i, year] - portfolio_values[i, year - 1] - annual_contribution) / ((1 + discount_rate) ** year))

    # Plotting future value results
    st.subheader("Monte Carlo Portfolio Forecast - Future Value")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i in range(min(simulations, 100)):
        ax1.plot(range(years + 1), portfolio_values[i], color='blue', alpha=0.1)
    ax1.set_title('Monte Carlo Portfolio Forecast Over Time - Future Value')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Future Portfolio Value ($)')
    ax1.grid(True)
    st.pyplot(fig1)

    # Plotting NPV results
    st.subheader("Monte Carlo Portfolio Forecast - Net Present Value (NPV)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(min(simulations, 100)):
        ax2.plot(range(years + 1), npv_over_time[i], color='green', alpha=0.1)
    ax2.set_title('Monte Carlo Portfolio Forecast Over Time - Net Present Value')
    ax2.set_xlabel('Years')
    ax2.set_ylabel('Net Present Value ($)')
    ax2.grid(True)
    st.pyplot(fig2)

    # Calculating statistics for future value
    total_ending_values = portfolio_values[:, -1]
    mean_ending_value = np.mean(total_ending_values)
    percentile_25 = np.percentile(total_ending_values, 25)
    percentile_75 = np.percentile(total_ending_values, 75)

    st.write(f"**Mean Ending Portfolio Value**: ${mean_ending_value:,.2f}")
    st.write(f"**25th Percentile Ending Portfolio Value**: ${percentile_25:,.2f}")
    st.write(f"**75th Percentile Ending Portfolio Value**: ${percentile_75:,.2f}")

    # Calculating statistics for net present value
    npv_values = npv_over_time[:, -1]
    mean_npv = np.mean(npv_values)
    npv_percentile_25 = np.percentile(npv_values, 25)
    npv_percentile_75 = np.percentile(npv_values, 75)

    st.write(f"**Mean Net Present Value**: ${mean_npv:,.2f}")
    st.write(f"**25th Percentile Net Present Value**: ${npv_percentile_25:,.2f}")
    st.write(f"**75th Percentile Net Present Value**: ${npv_percentile_75:,.2f}")

