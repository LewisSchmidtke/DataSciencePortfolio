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







