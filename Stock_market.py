import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('saketk511/2019-2024-us-stock-market-data', path='data/', unzip=True)

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
    ax = sns.lineplot(data=df, x="Date", y=company_name, linewidth=2, label=company_name.replace("_", " ") + " Price", color = rgb_color)
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
