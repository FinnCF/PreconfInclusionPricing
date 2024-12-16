import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import fftpack
from scipy.signal import find_peaks
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Set the style globally at the beginning
plt.rc('font', family='serif', size=11)

# Define color scheme
COLORS = {
    'primary': '#000000',      # Black
    'secondary': '#1f3c88',    # Dark Blue
    'tertiary': '#4a4e69',     # Dark Grey
    'accent': '#6c757d',       # Medium Grey
    'light': '#e9ecef'         # Light Grey
}

def apply_classic_style(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.grid(False)
    ax.set_facecolor('white')  # Ensure the background of the chart area is white


def logarithmic_model(x, a, b):
    """Logarithmic model function."""
    return a * np.log(b * x + 1)  # Adding 1 to avoid log(0)


def plot_histogram(data, title, xlabel, ylabel, threshold=None, threshold_label=None):
    """Plot histogram with optional threshold line."""
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=150, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
    
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', label=threshold_label)
        plt.legend()

    plt.title(title, fontsize=12, color=COLORS['primary'])
    plt.xlabel(xlabel, fontsize=11, color=COLORS['primary'])
    plt.ylabel(ylabel, fontsize=11, color=COLORS['primary'])
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def find_logarithmic_parameters(df):
    """
    Analyze cumulative rewards from transaction data and fit a logarithmic model.
    
    Args:
    - df: DataFrame containing transaction data with 'block_number', 'gas_used', and 'priority_fee_per_gas'.
    """
    # Group by 'block_number' and calculate the cumulative gas used
    total_gas_used_df = df.groupby('block_number', as_index=False)['gas_used'].sum()
    total_gas_used_df.rename(columns={'gas_used': 'total_gas_used'}, inplace=True)
    df = df.merge(total_gas_used_df, on='block_number')
    df.sort_values(by=['block_number', 'index'], inplace=True)    
    df['cumulative_gas'] = df.groupby('block_number')['gas_used'].cumsum()
    df['proposer_reward'] = ((df['transaction_proposer_reward']))
    df['proposer_reward'] = df['proposer_reward'] / 1e18 # Convert to ETH
    df['cumulative_rewards'] = df.groupby('block_number')['proposer_reward'].cumsum()

    # Calculate the IQR
    block_rewards = df.groupby('block_number')['proposer_reward'].sum()        
    Q1 = block_rewards.quantile(0.25)
    Q3 = block_rewards.quantile(0.75)
    IQR = Q3 - Q1
    print(f"IQR: {IQR}")

    # Find the 1.5x IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"1.5x IQR Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # Identify outliers based on 1.5x IQR
    valid_blocks = block_rewards[(block_rewards >= lower_bound) & (block_rewards <= upper_bound)].index    
    outlier_blocks = block_rewards[(block_rewards < lower_bound) | (block_rewards > upper_bound)].index
    df_invalid = df[df['block_number'].isin(outlier_blocks)]
    df = df[df['block_number'].isin(valid_blocks)]
    total_proposer_reward_summary = df['proposer_reward'].groupby(df['block_number']).sum()
    print(total_proposer_reward_summary.describe())

    # Check for NaN or inf values in 'cumulative_gas' and 'cumulative_rewards'
    if df['cumulative_gas'].isnull().any() or df['cumulative_rewards'].isnull().any():
        print("NaN values found in cumulative_gas or cumulative_rewards.")
    if np.isinf(df['cumulative_gas']).any() or np.isinf(df['cumulative_rewards']).any():
        print("Inf values found in cumulative_gas or cumulative_rewards.")

    # Remove rows with NaN or inf values
    df = df.dropna(subset=['cumulative_gas', 'cumulative_rewards'])
    df = df[~np.isinf(df['cumulative_gas']) & ~np.isinf(df['cumulative_rewards'])]

    # Fit the logarithmic model
    popt, _ = curve_fit(logarithmic_model, df['cumulative_gas'], df['cumulative_rewards'], maxfev=10000)
    x_fit = np.linspace(df['cumulative_gas'].min(), df['cumulative_gas'].max(), 100)
    y_fit = logarithmic_model(x_fit, *popt)
    y_actual = df['cumulative_rewards']  # Actual cumulative rewards
    residuals = y_actual - logarithmic_model(df['cumulative_gas'], *popt)  # Use the model to get fitted values
    ss_res = np.sum(residuals**2)  # Sum of squares of residuals
    ss_tot = np.sum((y_actual - np.mean(y_actual))**2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot) 

    print('R^2: ', r_squared)
    print(popt)
    print(f"Logarithmic Model Parameters: a = {popt[0]:.4f}, b = {popt[1]:.4f}")
    print(f"Logarithmic Model: y = {popt[0]:.4f} * log({popt[1]:.4f} * x + 1)")

    # Plotting cumulative rewards vs cumulative gas and the logarithmic fit
    plt.figure(figsize=(12, 6))
    plt.scatter(df['cumulative_gas'], df['cumulative_rewards'], linestyle='-', color='grey', alpha=0.01, label='Data Points')
    plt.axvline(x=1.5e7, color='black', linestyle='--', label='Target Gas Usage')  # New line added
    plt.plot(x_fit, y_fit, color='black', linewidth=2, label='Logarithmic Fit')
    plt.title('Cumulative Rewards vs. Cumulative Gas with Logarithmic Fit', fontsize=12, color=COLORS['primary'])
    plt.xlabel('Cumulative Gas', fontsize=11, color=COLORS['primary'])
    plt.ylabel('Cumulative Rewards (ETH)', fontsize=11, color=COLORS['primary'])
    plt.grid(True, linestyle='--', alpha=0, color=COLORS['tertiary'])
    plt.legend()
    plt.tight_layout()
    plt.show()

    return popt  

def add_builder_rewards_and_proposer_rewards(df):
    """
    Add builder payment and proposer columns from the last transaction of each block - according to Methodology.
    """
    # First convert 'value' column to numeric type
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['total_priority_fee'] = df['priority_fee_per_gas'] * df['gas_used']
    
    # Discovering if MEV-Boost, and the builder payment/proposer/builder
    for block_number, group in df.groupby('block_number'):
        if not group.empty:

            # Observing if we are looking at MEV-Boost
            sorted_group = group.sort_values('index')
            last_index = sorted_group.index[-1]
            to = df.loc[last_index]['to']
            _from = df.loc[last_index]['from']

            # Mev Boost
            if(_from == df.loc[last_index]['builder']):
                payment = df.loc[last_index]['value']
                mev_boost_builder = df.loc[last_index]['builder']
                
                # Calculate builder reward as the sum of total priority fees and payment to the builder
                df.loc[group.index, 'proposer'] = to  # Assign proposer to all rows in this block
                df.loc[group.index, 'proposer_payment'] = payment  # Assign proposer payment to all rows in this block

                # Adding the specific builder rewards from that transaction
                for index in group.index[:-1]:  # Exclude the last index
                    if(df.loc[index, 'to'] == mev_boost_builder):
                        df.loc[index, 'transaction_builder_reward'] = df.loc[index, 'total_priority_fee'] + df.loc[index, 'value']
                    else:
                        df.loc[index, 'transaction_builder_reward'] = df.loc[index, 'total_priority_fee'] 

    for block_number, group in df.groupby('block_number'):
        
         # Observing if we are looking at MEV-Boost
        sorted_group = group.sort_values('index')
        last_index = sorted_group.index[-1]
        to = df.loc[last_index]['to']
        _from = df.loc[last_index]['from']
        
        if not group.empty:     
                total_builder_reward = group['transaction_builder_reward'].sum()  # Sum of builder rewards
                df.loc[group.index, 'blocks_builder_reward'] = total_builder_reward  # Assign to all rows in the group

    for block_number, group in df.groupby('block_number'):
        
        # Observing if we are looking at MEV-Boost
        sorted_group = group.sort_values('index')
        last_index = sorted_group.index[-1]
        to = df.loc[last_index]['to']
        _from = df.loc[last_index]['from']

        if not group.empty:
            if(_from == df.loc[last_index]['builder']):
                for index in group.index[:-1]:  # MEV 
                    df.loc[index, 'transaction_proposer_reward'] = df.loc[index, 'transaction_builder_reward'] * (df.loc[index, 'proposer_payment'] / df.loc[index, 'blocks_builder_reward'])
            else:
                for index in group.index:  # Traditional
                    df.loc[index, 'transaction_proposer_reward'] = df.loc[index, 'total_priority_fee'] 

    for block_number, group in df.groupby('block_number'):
        if not group.empty:  
                total_proposer_reward = group['transaction_proposer_reward'].sum()  # Sum of proposer rewards
                df.loc[group.index, 'blocks_proposer_reward'] = total_proposer_reward  # Assign to all rows in the group

    # Verify the calculations
    block_summary = df.groupby('block_number').agg({
        'blocks_proposer_reward': 'first'
    }).dropna()
    
    print(f"Mean sum of proposer rewards: {block_summary['blocks_proposer_reward'].mean()}")
    return df  # Ensure the modified DataFrame is returned

def validate_autocorrelation(df):
    """
    Validate the autocorrelation of variables that contribute to the proposer reward.
    """
    # For each 'block_number', get the previous 20 blocks and make a large list of their bottom 3 fees
    for block_number in df['block_number'].unique():
        previous_block_numbers = df[df['block_number'] < block_number]['block_number'].unique()[-20:]
        previous_blocks = df[df['block_number'].isin(previous_block_numbers)]
        bottom_3_fees_list = []
        for _, group in previous_blocks.groupby('block_number'):
            filtered_group = group[group['priority_fee_per_gas'] >= 2] # As in GETH
            bottom_3_fees = filtered_group['priority_fee_per_gas'].nsmallest(3).tolist() # Lowest 3 fees
            bottom_3_fees_list.extend(bottom_3_fees) 
        if bottom_3_fees_list:
            percentile_value = np.percentile(bottom_3_fees_list, 60)
            df.loc[df['block_number'] == block_number, 'bottom_3_fees'] = percentile_value
        else:
            df.loc[df['block_number'] == block_number, 'bottom_3_fees'] = np.nan
            print(f"Block {block_number}: No valid fees found, assigned NaN")
    plt.figure(figsize=(12, 6))
    plt.plot(df['block_number'], df['bottom_3_fees'] / 1e9, color='black', label='60th Percentile of Bottom 3 Fees')
    plt.title('Block Number vs. 20 block bottom 3 fees list 60th percentile')
    plt.xlabel('Block Number')
    plt.ylabel('60th Percentile of Bottom 3 Fees (GWEI)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



    # Group by block_number and aggregate both metrics
    aggregated_df = df.groupby('block_number').agg({
        'blocks_proposer_reward': 'first',
        'priority_fee_per_gas': 'median'
    }).reset_index()

    # Calculate averages for both metrics
    avg_reward = aggregated_df['blocks_proposer_reward'].mean()
    avg_fee = aggregated_df['priority_fee_per_gas'].mean()

    # Plot both metrics over block range
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Proposer Reward Plot
    ax1.plot(aggregated_df['block_number'], aggregated_df['blocks_proposer_reward'] / 1e18, color='black', label='Block Proposer Reward')
    ax1.axhline(y=avg_reward / 1e18, color='red', linestyle='--', label='Average Block Proposer Reward')
    ax1.set_title('Block Proposer Rewards Over Time')
    ax1.set_xlabel('Block Number')
    ax1.set_ylabel('Block Proposer Reward (ETH)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Priority Fee Plot
    ax2.plot(aggregated_df['block_number'], aggregated_df['priority_fee_per_gas'] / 1e9, color='black', label='Median Priority Fee')
    ax2.axhline(y=avg_fee / 1e9, color='red', linestyle='--', label='Average Priority Fee')
    ax2.set_title('Priority Fees Over Time')
    ax2.set_xlabel('Block Number')
    ax2.set_ylabel('Priority Fee (GWEI)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Process both metrics separately
    metrics = {
        'blocks_proposer_reward': ('Block Proposer Reward (ETH)', 1e18),
        'priority_fee_per_gas': ('Priority Fee (GWEI)', 1e9)
    }

    for metric, (ylabel, scale) in metrics.items():
        # Detrending
        X = sm.add_constant(aggregated_df['block_number'])
        y = aggregated_df[metric]
        model_trend_ols = sm.OLS(y, X).fit()
        
        # Print regression results
        print(f"\nRegression Results for {metric}:")
        print("================================")
        print(f"R-squared: {model_trend_ols.rsquared:.4f}")
        print(f"Coefficient (slope): {model_trend_ols.params[1]:.4e}")
        print(f"Intercept: {model_trend_ols.params[0]:.4e}")
        print(f"P-values: {model_trend_ols.pvalues}")
        aggregated_df[f'trend_{metric}'] = model_trend_ols.predict(X)
        aggregated_df[f'detrended_{metric}'] = aggregated_df[metric] - aggregated_df[f'trend_{metric}']

        # Plot original and detrended data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(aggregated_df['block_number'], aggregated_df[metric] / scale, label='Original Data', color='grey')
        ax1.plot(aggregated_df['block_number'], aggregated_df[f'trend_{metric}'] / scale, label='Fitted Linear Trend', color='black')
        ax1.set_title(f'{metric.replace("_", " ").title()} with Trend')
        ax1.set_xlabel('Block Number')
        ax1.set_ylabel(ylabel)
        ax1.legend()
        ax1.grid(False)

        ax2.plot(aggregated_df['block_number'], aggregated_df[f'detrended_{metric}'] / scale, label='Detrended Data', color='black')
        ax2.set_title(f'Detrended {metric.replace("_", " ").title()}')
        ax2.set_xlabel('Block Number')
        ax2.set_ylabel(f'Detrended {ylabel}')
        ax2.legend()
        ax2.grid(False)
        
        plt.tight_layout()
        plt.show()

        # Perform spectral analysis on detrended data
        perform_spectral_analysis(aggregated_df[f'detrended_{metric}'], metric)

        # 1. ACF and PACF plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        plot_acf(aggregated_df[f'detrended_{metric}'], ax=ax1, lags=60, color='black')
        ax1.set_title(f'Autocorrelation Function for Median {metric.replace("_", " ").title()}')
        ax1.grid(False)
        
        plot_pacf(aggregated_df[f'detrended_{metric}'], ax=ax2, lags=20, color='black')
        ax2.set_title(f'Partial Autocorrelation Function for Median {metric.replace("_", " ").title()}')
        ax2.grid(False)
        
        plt.tight_layout()
        plt.show()

        # 2. Perform ADF test
        adf_result = adfuller(aggregated_df[f'detrended_{metric}'])
        print(f'\nAugmented Dickey-Fuller Test Results for {metric}:')
        print(f'ADF Statistic: {adf_result[0]}')
        print(f'p-value: {adf_result[1]}')
        print('Critical values:')
        for key, value in adf_result[4].items():
            print(f'\t{key}: {value}')

        # 3. Fit ARIMA model
        # Start with a simple AR(1) model as baseline
        model = ARIMA(aggregated_df[f'detrended_{metric}'], order=(1, 0, 0))
        results = model.fit()
        
        # Plot the original data vs predictions
        plt.figure(figsize=(12, 6))
        plt.plot(aggregated_df['block_number'], 
                aggregated_df[f'detrended_{metric}'] / scale, 
                label='Detrended Data', 
                color='grey', 
                alpha=0.6)
        plt.plot(aggregated_df['block_number'], 
                results.fittedvalues / scale, 
                label='AR(1) Predictions', 
                color='black', 
                linewidth=2)
        plt.title(f'AR(1) Model Fit for {metric.replace("_", " ").title()}')
        plt.xlabel('Block Number')
        plt.ylabel(f'Detrended {ylabel}')
        plt.legend()
        plt.grid(False)
        plt.show()

        # 4. Print model summary
        print(f'\nAR(1) Model Summary for {metric}:')
        print(results.summary().tables[1])

        # 5. Analyze residuals
        residuals = results.resid
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Residuals over time
        ax1.plot(aggregated_df['block_number'], residuals / scale, color='black')
        ax1.set_title(f'Model Residuals for {metric.replace("_", " ").title()}')
        ax1.set_xlabel('Block Number')
        ax1.set_ylabel(f'Residual {ylabel}')
        ax1.grid(False)

        # Residuals histogram
        ax2.hist(residuals / scale, bins=50, color='black', alpha=0.6)
        ax2.set_title('Histogram of Residuals')
        ax2.set_xlabel(f'Residual {ylabel}')
        ax2.set_ylabel('Frequency')
        ax2.grid(False)
        
        plt.tight_layout()
        plt.show()

        # 6. Calculate and print prediction intervals
        mean_prediction = results.fittedvalues.mean() / scale
        std_prediction = results.fittedvalues.std() / scale
        print(f'\nPrediction Summary for {metric}:')
        print(f'Mean predicted value: {mean_prediction:.6f}')
        print(f'Standard deviation: {std_prediction:.6f}')
        print(f'95% Prediction Interval: ({mean_prediction - 1.96*std_prediction:.6f}, {mean_prediction + 1.96*std_prediction:.6f})')

def perform_spectral_analysis(detrended_data, metric_name):
    """
    Perform spectral analysis on the detrended data and plot in black and white.
    """
    # Apply Fourier transform
    fft_result = fftpack.fft(detrended_data.values)

    # Compute power spectrum and corresponding frequency vector
    power_spectrum = np.abs(fft_result)**2
    frequencies = fftpack.fftfreq(len(detrended_data), d=1)

    # Keep only the positive part of the spectrum
    power_spectrum = power_spectrum[:len(detrended_data)//2]
    frequencies = frequencies[:len(detrended_data)//2]

    # Convert frequencies to periods and express them in blocks
    periods = 1 / frequencies 

    # Identify the two periods that have the largest powers
    peaks, _ = find_peaks(power_spectrum, distance=100)
    top_periods = periods[peaks]

    # Plot the power spectrum in black and white
    plt.figure(figsize=(10, 6))
    plt.stem(periods, power_spectrum, '.', label='Power', linefmt='grey', markerfmt='ko', basefmt='k-')
    plt.plot(top_periods, power_spectrum[peaks], 'k^', label='Maxima')
    plt.xlabel('Period (blocks)')
    plt.ylabel('Power')
    plt.title(f'Power Spectrum of Detrended {metric_name.replace("_", " ").title()}')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()

# Pricing Function
def price_with_function(tx_size_gas, preconfed_gas_used, popt):
    return logarithmic_model(30_000_000 - preconfed_gas_used, popt[0], popt[1]) - logarithmic_model(30_000_000 - preconfed_gas_used - tx_size_gas, popt[0], popt[1])

def constant_price_function(tx_size_gas, inclusion_tip_per_gas):
    """Calculate price as a constant multiplier of transaction size."""
    return tx_size_gas * inclusion_tip_per_gas / 1e18  # Adjust for millions of gas and scale to ETH

def plot_pricing_and_rewards(popt):
    # Create meshgrid for the surface with higher granularity
    tx_sizes = np.linspace(0, 30_000_000, 300)  # Increased from 100 to 300 points
    preconfed_sizes = np.linspace(0, 30_000_000, 300)  # Increased from 100 to 300 points
    X, Y = np.meshgrid(tx_sizes, preconfed_sizes)    
    Z = np.zeros_like(X)
    for i in range(len(tx_sizes)):
        for j in range(len(preconfed_sizes)):
            # Only calculate price if total gas doesn't exceed 30M
            if X[i,j] + Y[i,j] <= 30_000_000:
                Z[i,j] = (price_with_function(X[i,j], Y[i,j], popt) * 1e9) / X[i,j]
            else:
                Z[i,j] = np.nan  # Set invalid combinations to NaN    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')    
    surf = ax.plot_surface(X/1e6, Y/1e6, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Transaction Size (M Gas)')
    ax.set_ylabel('Pre-confirmed Gas (M Gas)')
    ax.set_zlabel('Average Inclusion Tip Per Gas (GWEI)')
    ax.set_title('Pricing Surface')
    ax.set_zlim(0, 16)  # Limit the z-axis to 20
    fig.colorbar(surf, ax=ax, label='Price (ETH)')
    plt.tight_layout()
    plt.show()
    print('Sample Price: ', price_with_function(10_000_000, 0, popt))
    
    x_values = np.linspace(0, 30_000_000, 300)  # Define the range for x
    y_values = [price_with_function(x, 0, popt) if x != 0 else np.nan for x in x_values]  # Calculate price per gas
    y_values_constant_0_5_gwei = [constant_price_function(x, 0.5e9) for x in x_values]  # Calculate price per gas   
    y_values_constant_1_gwei  = [constant_price_function(x, 1e9) for x in x_values]  # Calculate price per gas   
    y_values_constant_1_5_gwei  = [constant_price_function(x, 1.5e9) for x in x_values]  # Calculate price per gas   
    plt.plot(x_values / 1e6, y_values, color='black', label='Models Proposer Reward (GWEI)')
    plt.plot(x_values / 1e6, y_values_constant_1_5_gwei, color='black', linestyle='--', label='Constant Inclusion Tip Proposer Reward (1.5 GWEI)')
    plt.plot(x_values / 1e6, y_values_constant_1_gwei, color='gray', linestyle='--',  label='Constant Inclusion Tip Proposer Reward (1 GWEI)')
    plt.plot(x_values / 1e6, y_values_constant_0_5_gwei, color='darkgray',  linestyle='--', label='Constant Inclusion Tip Proposer Reward (0.5 GWEI)')
    plt.xlabel('Inclusion Preconfirmations (M Gas)')
    plt.ylabel('Proposer Reward (ETH)')
    plt.title('Proposer Reward vs Gas Used for Inclusion Preconfirmations')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Difference Plot
    differences = {
        '0.5 GWEI': [y_const - y for y, y_const in zip(y_values, y_values_constant_0_5_gwei)],
        '1 GWEI': [y_const - y for y, y_const in zip(y_values, y_values_constant_1_gwei)],
        '1.5 GWEI': [y_const - y for y, y_const in zip(y_values, y_values_constant_1_5_gwei)],
        'Model': [0 for _ in y_values]  
    }
    plt.figure(figsize=(12, 6))
    color_map = {
        '0.5 GWEI': 'gray',
        '1 GWEI': 'darkgray',
        '1.5 GWEI': 'black',
        'Model': 'black'
    }
    for label, y_diff in differences.items():
        if label != 'Model':
            plt.plot(x_values / 1e6, y_diff, linestyle='--', color=color_map[label], linewidth=1.5, label=f'Difference (Constant {label} - Model)')
    # Handle the 'Model' line separately
    plt.plot(x_values / 1e6, differences['Model'], color=color_map['Model'], linewidth=1.5, label='Model')
    plt.xlabel('Inclusion Preconfirmations (M Gas)')
    plt.ylabel('Difference in Proposer Reward (ETH)')
    plt.title('Difference in Proposer Reward vs Gas Used for Inclusion Preconfirmations')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))    
    x = np.linspace(0, 30_000_000, 1000)
    y = logarithmic_model(x, *popt)
    plt.plot(x/1e6, y, 'k-', label='Cumulative Reward Curve')    
    x_preconfed = 17_000_000
    y_preconfed = logarithmic_model(x_preconfed, *popt)
    plt.plot([x_preconfed/1e6, x_preconfed/1e6], [0, y_preconfed], 'gray', linestyle='--', label='Limit (30M) - Preconfirmed (13M)')
    plt.plot([0, x_preconfed/1e6], [y_preconfed, y_preconfed], 'gray', linestyle='--')
    x_tx = 15_000_000
    y_tx = logarithmic_model(x_tx, *popt)
    plt.plot([x_tx/1e6, x_tx/1e6], [0, y_tx], 'blue', linestyle='--', label='Limit (30M) - Preconfirmed (13M) -Transaction (2M)')
    plt.plot([0, x_tx/1e6], [y_tx, y_tx], 'blue', linestyle='--')
    plt.xlabel('Gas Used (M)')
    plt.ylabel('Cumulative Reward (ETH)')
    plt.title('Inclusion Pricing Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():

    # Load transaction data and proposer
    df = pd.read_csv('./Data/Transactions.csv')   
    df_builder = pd.read_csv('./Data/Builders.csv')
    df_builder.rename(columns={'number': 'block_number'}, inplace=True)    
    df = df.merge(df_builder, on='block_number', how='left')  # Adding the proposer in
    df.rename(columns={'miner': 'builder'}, inplace=True)

    # Adding proposer rewards
    df = add_builder_rewards_and_proposer_rewards(df)

    # Running analyzing cumulative rewards
    popt = find_logarithmic_parameters(df)

    # Plotting the pricing and rewards
    plot_pricing_and_rewards(popt)

    # Running AR Analysis
    validate_autocorrelation(df)
    
if __name__ == "__main__":
    main()
