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

# Run Monte Carlo Simulation
run_simulation = st.button("Run Simulation")

if run_simulation:
    # Set the random seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # Monte Carlo simulations in chunks to save memory
    chunk_size = 100  # Process simulations in chunks to reduce memory usage
    portfolio_values_chunks = []
    npv_over_time_chunks = []

    for chunk_start in range(0, simulations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, simulations)
        current_chunk_size = chunk_end - chunk_start

        # Initialize chunk arrays
        portfolio_chunk = np.empty((current_chunk_size, years + 1))
        portfolio_chunk[:, 0] = initial_investment
        npv_chunk = np.zeros((current_chunk_size, years + 1))

        # Run simulations for the current chunk
        for i in range(current_chunk_size):
            for year in range(1, years + 1):
                if year == 1:
                    random_return = rng.normal(average_return, std_dev)
                else:
                    # Correlate current year's movement with the previous year
                    if portfolio_chunk[i, year - 1] > portfolio_chunk[i, year - 2]:
                        # Previous year was up
                        random_return = rng.normal(average_return, std_dev) if rng.random() < 0.7 else rng.normal(-average_return, std_dev)
                    else:
                        # Previous year was down
                        random_return = rng.normal(-average_return, std_dev) if rng.random() < 0.7 else rng.normal(average_return, std_dev)

                # Update portfolio value with annual contribution and return
                portfolio_chunk[i, year] = (portfolio_chunk[i, year - 1] + annual_contribution) * (1 + random_return)

                # Calculate NPV over time
                if year == 1:
                    npv_chunk[i, year] = initial_investment / ((1 + discount_rate) ** 0)  # Discount initial investment (which is itself)
                else:
                    npv_chunk[i, year] = npv_chunk[i, year - 1] + (annual_contribution / ((1 + discount_rate) ** year)) + \
                                         ((portfolio_chunk[i, year] - portfolio_chunk[i, year - 1] - annual_contribution) / ((1 + discount_rate) ** year))

        # Append the results of the current chunk
        portfolio_values_chunks.append(portfolio_chunk)
        npv_over_time_chunks.append(npv_chunk)

    # Concatenate all chunks to form the complete arrays
    portfolio_values = np.concatenate(portfolio_values_chunks, axis=0)
    npv_over_time = np.concatenate(npv_over_time_chunks, axis=0)

    # Plotting future value results
    st.subheader("Monte Carlo Portfolio Forecast - Future Value")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i in range(0, simulations, max(1, simulations // 10)):
        ax1.plot(range(years + 1), portfolio_values[i], color='blue', alpha=0.1)
    ax1.set_title('Monte Carlo Portfolio Forecast Over Time - Future Value')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Future Portfolio Value ($)')
    ax1.grid(True)
    st.pyplot(fig1)

    # Plotting NPV results
    st.subheader("Monte Carlo Portfolio Forecast - Net Present Value (NPV)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(0, simulations, max(1, simulations // 10)):
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

