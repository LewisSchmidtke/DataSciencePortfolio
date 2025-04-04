import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from kaggle.api.kaggle_api_extended import KaggleApi

# api = KaggleApi()
# api.authenticate()
# api.dataset_download_files('saketk511/2019-2024-us-stock-market-data', path='data/', unzip=True)

# Extract data
df = pd.read_csv('data/Stock Market Dataset.csv')
df = df.iloc[::-1] # Flip dataframe so data is chronological
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")  # Convert to datetime


# Tech Stocks with correct color
companies_with_color = [("Tesla_Price", (139,0,139)), ("Apple_Price",	(0, 0, 0)), ('Microsoft_Price', (114, 114, 114)),
                        ('Google_Price', (251, 188, 4)), ('Nvidia_Price', (118, 185, 0)),('Netflix_Price', (229, 9, 20)),
                        ('Amazon_Price', (255, 153, 0)),('Meta_Price', (0, 129, 251))]

# ---------------------------------- TECH STOCKS relative to each other ---------------------------------------------
sns.set_style('whitegrid')
plt.figure(figsize = (18,8))

for company in companies_with_color:
    company_name, company_color = company
    rgb_color = [value / 255 for value in company_color]
    df[company_name] = df[company_name].rolling(window=6, min_periods=1).mean()
    ax = sns.lineplot(data=df, x="Date", y=company_name, linewidth=2, label=company_name.replace("_", " ") ,
                      color = rgb_color)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))  # Displaying Quarters on X-Axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xticks(rotation=45)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Stock Price", fontsize=14)
plt.title("Stock Price Rolling Average for the top 8 Tech Stocks", fontsize=18, fontweight="bold")
plt.legend(title="Company Stock Price")
plt.tight_layout()
plt.savefig("./Plots/StockPriceRollingAVG(Tech).png")
plt.show()

# ---------------------------------- Tech Stock Price vs Volume ------------------------------------------------

ROWS = 4
COLS = 2
idx_tracker = 0
fig, axs = plt.subplots(ROWS, COLS, figsize=(24, 20))
for r in range(ROWS):
    for c in range(COLS):
        # Extract Company Data
        company_price, company_color = companies_with_color[idx_tracker]
        rgb_color = [value / 255 for value in company_color]
        company_name = company_price.split("_")[0]
        company_vol = company_name + "_Vol."

        # Calculate smoothed volume curve
        df[company_vol] = df[company_vol].rolling(window=20, min_periods=1).mean()

        # Plot price on the primary y-axis
        axs[r, c].plot(df["Date"], df[company_price], color=rgb_color, label=company_name + ' Price', linewidth=3)
        axs[r, c].set_xlabel('Date')
        axs[r, c].set_ylabel(company_name + ' Stock Price ($)', color=rgb_color, fontsize=14, fontweight='bold')
        axs[r, c].tick_params(axis='y', labelcolor=rgb_color)

        # Create a second y-axis for trading volume
        ax2 = axs[r, c].twinx() # Copy x-axis from first plot
        ax2.plot(df["Date"], df[company_vol], color='g', alpha=0.75, label=company_name+ ' Volume', linewidth=2)
        ax2.set_ylabel(company_name + ' Trading Volume', color='g', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='g')

        # Create rotated x labels only showing first of each quarter to not clutter x-axis
        axs[r, c].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
        axs[r, c].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[r, c].tick_params(axis='x', rotation=45)

        # Set title
        plt.title(company_name + " Stock Price and Volume", fontsize=14, fontweight='bold')
        plt.tight_layout()
        idx_tracker += 1 # Update tracker
plt.savefig("./Plots/Price_vs_Volume_(Tech).png")
plt.show()

# ------------------------------ Volatility Calculation and Plotting --------------------------------------

# Calculate daily returns for bitcoin, ethereum, Nasdaq and S&P 500

# Transform price data from 40,000.00 to 40000.00 to later apply pct_change function
df['Bitcoin_Price'] = df['Bitcoin_Price'].astype(str).str.replace(",", "")
df['Bitcoin_Price'] = df['Bitcoin_Price'].astype(float)
df['Ethereum_Price']= df['Ethereum_Price'].astype(str).str.replace(",", "")
df['Ethereum_Price']= df['Ethereum_Price'].astype(float)
df['Nasdaq_100_Price']= df['Nasdaq_100_Price'].astype(str).str.replace(",", "")
df['Nasdaq_100_Price']= df['Nasdaq_100_Price'].astype(float)
df['S&P_500_Price']= df['S&P_500_Price'].astype(str).str.replace(",", "")
df['S&P_500_Price']= df['S&P_500_Price'].astype(float)

# Calculate daily return for the 4 objects
df['Bitcoin_Return'] = df['Bitcoin_Price'].pct_change() # pct_change = (value_i - value_i-1) / value_i-1
df['Ethereum_Return'] = df['Ethereum_Price'].pct_change()
df['Nasdaq_Return'] = df['Nasdaq_100_Price'].pct_change()
df['SP500_Return'] = df['S&P_500_Price'].pct_change()

# Calculate rolling volatility (30-day window)
df['Bitcoin'] = df['Bitcoin_Return'].rolling(window=30).std() * np.sqrt(30)
df['Ethereum'] = df['Ethereum_Return'].rolling(window=30).std() * np.sqrt(30)
df['Nasdaq'] = df['Nasdaq_Return'].rolling(window=30).std() * np.sqrt(30)
df['S&P500'] = df['SP500_Return'].rolling(window=30).std() * np.sqrt(30)


volatility_long_df = df[['Bitcoin', 'Ethereum', 'Nasdaq', 'S&P500']]
volatility_long_df = volatility_long_df.melt(var_name='Asset', value_name='Volatility')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 8))
sns.violinplot(x='Asset', y='Volatility', data=volatility_long_df, hue="Asset", palette="rocket", legend=False)

sns.despine(left=True, bottom=True)
plt.title('Volatility Distribution for Bitcoin, Ethereum, Nasdaq, and S&P 500', fontsize=16, fontweight='bold', loc="left")
plt.ylabel('Volatility')
plt.yticks(np.arange(0, 0.8, 0.05)) # adds more y ticks

plt.xlabel("")
plt.tight_layout()
plt.savefig("./Plots/Volatility_Tech+Crypto.png")
plt.show()

# -------------------------- Tech Share in S&P 500 ------------------------------------------------

# Cumulative price for each entry of the big 8 tech stock
df["Tech Stock Prices"] =df[["Tesla_Price", "Apple_Price", "Microsoft_Price",
                             "Google_Price", "Nvidia_Price", "Netflix_Price", "Amazon_Price", "Meta_Price"]].sum(axis=1)
# Calculate what percentage of the SP500 Price is made up of the big 8 tech stocks
df["Percentage_Top8_SP500"] = (df["Tech Stock Prices"] / df["S&P_500_Price"]) * 100

df["Tech_Stock_Prices_Rolling"] = df["Tech Stock Prices"].rolling(window=10, min_periods=1).mean()
df["Percentage_Top8_SP500_Rolling"] = df["Percentage_Top8_SP500"].rolling(window=10, min_periods=1).mean()
df["S&P_500_Price_Rolling"] = df["S&P_500_Price"].rolling(window=10, min_periods=1).mean()

sns.set_style("white")
# Create a single figure and axis
fig, ax1 = plt.subplots(figsize=(14, 6))

# First y-axis (left) for stock prices
sns.lineplot(ax=ax1, data=df, x="Date", y="Tech_Stock_Prices_Rolling", color="#39FF14", label="Main 8 Tech Stock Prices", linewidth=3)
sns.lineplot(ax=ax1, data=df, x="Date", y="S&P_500_Price_Rolling", color="#4A0072", label="S&P 500 Price", linewidth=3)
ax1.set_ylabel("Stock Prices", fontsize=14, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Second y-axis (right) for percentage
ax2 = ax1.twinx()
sns.lineplot(ax=ax2, data=df, x="Date", y="Percentage_Top8_SP500_Rolling", linewidth=2, color="#6D8196", label="% of S&P 500 Market Value", linestyle="--", markevery=30)
ax2.set_ylabel("Percentage %", fontsize=14, fontweight='bold', color='#6D8196')
ax2.tick_params(axis='y', labelcolor='#6D8196')

# Set x-axis formatting
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], bymonthday=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.tick_params(axis='x', rotation=45)

# Title and x-axis label
ax1.set_title("Silicon Valley's Market Footprint: Tech Prices and S&P 500 Composition", fontsize=18, fontweight='bold')
ax1.set_xlabel("Date", fontsize=14, fontweight='bold')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

# Remove the default legend from the second axis
ax2.get_legend().remove()

plt.tight_layout()
plt.savefig("./Plots/S&P500_Market_Composition.png")
plt.show()

# -------------------------- Risk Return Scatter Plot ------------------------------------------------

# Get all relevant prices
resource_columns = ["Copper_Price","Platinum_Price","Silver_Price","Gold_Price",]
company_stocks = [value[0] for value in companies_with_color]
company_stocks.append("Berkshire_Price")
analyze_columns = resource_columns + company_stocks

# Fix prices for all price columns
for column in analyze_columns:
        df[column] = df[column].astype(str).str.replace(",", "")
        df[column] = df[column].astype(float)

# Calculation for risk and return for a given window
window_months = 6
window_days = window_months * 30 #approx
results = {
        'company': [],
        'return': [],
        'risk': [],
        'end_date':[],
        'start_date':[],
        'period': [],
}
# Create a period column for grouping (year and half: '2018H1', '2018H2', etc.)
df['year'] = df["Date"].dt.year
df['month'] = df["Date"].dt.month
df['half'] = (df['month'] > 6).astype(int) + 1  # 1 for Jan-Jun, 2 for Jul-Dec
df['period'] = df['year'].astype(str) + 'H' + df['half'].astype(str)

for stock in analyze_columns:
    grouped = df.groupby([column])

    for period, group in grouped:
        group[f"{stock}_return"] = group[stock].pct_change()
        start_price = group[stock].iloc[0]
        end_price = group[stock].iloc[-1]
        # Return and risk for 6-month period in percent
        period_return = ((end_price / start_price) - 1) * 100
        period_risk = (group[f"{stock}_return"].std() * np.sqrt(len(group))) * 100
        start_date = group["Date"].min()
        end_date = group["Date"].max()

        results['company'].append(stock)
        results['period'].append(period)
        results['start_date'].append(start_date)
        results['end_date'].append(end_date)
        results['return'].append(period_return)
        results['risk'].append(period_risk)

# Define view limits for the main plot --> max x: 100% return, max y: 300% risk
main_view_limits = (100, 300)
result_df = pd.DataFrame(results)
# Create separate DF with values outside the main view limits return
outlier_df = result_df[result_df['return'] >= main_view_limits[0]]

sns.set_style("whitegrid")
fig, ax15 = plt.subplots(figsize=(14, 6))
scatter = sns.scatterplot(data=result_df, x="return", y="risk", hue="company", style="company",
                          alpha=0.8, palette="Dark2", s=200, ax=ax15)
# Set axis limits
ax15.set_xlim(-30, main_view_limits[0])
ax15.set_ylim(-1, main_view_limits[1])

# Add a reference line at x = 0 to mark negative returns
ax15.axvline(x=0, color='r', linestyle='-', alpha=0.3)
ax15.tick_params(axis='both', which='major', labelsize=18) # Increase tick sizes

if not outlier_df.empty:
    # The dimensions are [left, bottom, width, height] in figure coordinates
    inset_ax = fig.add_axes([0.7, 0.62, 0.25, 0.25])
    # Extract for color matching
    legend = ax15.get_legend()
    color_mapping = {}
    marker_mapping = {}

    if legend:
        for text, handle in zip(legend.get_texts(), legend.legend_handles):
            company = text.get_text()

            if hasattr(handle, 'get_marker'): # Marker
                marker_mapping[company] = handle.get_marker()
            if hasattr(handle, 'get_color'): # Color
                color_mapping[company] = handle.get_color()
            elif hasattr(handle, 'get_facecolor'):
                color_mapping[company] = handle.get_facecolor()
            elif hasattr(handle, 'get_facecolors'):
                color_mapping[company] = handle.get_facecolors()[0]

    # Plot outliers in the inset
    for company in outlier_df['company'].unique():
        company_data = outlier_df[outlier_df['company'] == company]

        # Get the color and marker for this stock
        color = color_mapping.get(company, 'gray')  # Default to gray if not found
        marker = marker_mapping.get(company, 'o')  # Default to circle if not found

        sns.scatterplot(
            x=company_data['return'],
            y=company_data['risk'],
            c=[color],  # Use the same color as main plot
            marker=marker,  # Use the same marker as main plot
            s=160,
            alpha=0.8,
            legend=False,
            ax=inset_ax,
        )

    # Configure inset axes
    inset_ax.set_title('Outliers (High Returns)', fontsize=14)
    inset_ax.grid(True, linestyle='--', alpha=0.6)
    # set inset axes limits based on the data
    inset_ax.set_ylim(0, outlier_df['risk'].max() * 1.1)
    inset_ax.set_xlim(main_view_limits[0], outlier_df['return'].max() * 1.1)
    inset_ax.set_xlabel('Asset Return (%)')
    inset_ax.set_ylabel('Asset Risk (%)')
    # box around the inset
    inset_ax.patch.set_edgecolor('black')
    inset_ax.patch.set_linewidth(1)

ax15.legend(loc="upper left")
ax15.set_title("Asset Risk-Return Comparison for 6 Month Periods",fontsize=18, fontweight='bold')
ax15.set_xlabel("Asset Return (%)", fontsize=14, fontweight='bold')
ax15.set_ylabel("Asset Risk (%)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./Plots/risk-return_comparison.png')
plt.show()

















