# DataScience and Machine Learning Portfolio
This repository is used to create visually appealing plots for various datasets to be displayed in this README. All relevant graphics can be found here, whereas some selected plots can also be found on my portfolio website.

## Data Overview
- 2019-2024 US Stock Market Data (ONGOING) ✔️
- Global Cybersecurity Threats 2015-2024 (ONGOING) ✔️
- Credit Card Fraud Detection (Planned)

## US Stock Market Data 2019-2024
Before applying machine learning approaches to different use cases in the dataset, I decide on finding key insights that help me better understand the data I am working with.
### 1. Tech Stock Price Rolling Average:
![image alt](https://github.com/LewisSchmidtke/DataSciencePortfolio/blob/main/Plots/StockPriceRollingAVG(Tech).png?raw=true)
The main idea was to first plot the stock price of the big 8 to compare any relative trends. It is clear that the overall trend for all stocks is roughly the same with dips at the beginning and end of the pandemic, overall growth during the pandemic from the "Tech Bubble" and post-pandemic growth from the AI-Boom.
### 2. Tech Stock Price vs Trade Volume:
![image alt](https://github.com/LewisSchmidtke/DataSciencePortfolio/blob/main/Plots/Price_vs_Volume_(Tech).png?raw=true)
Secondly I wanted to compare the individual tech prices with their trading volume, to analyze trading patterns. The pattern visible is that almost all tech stocks have been traded massively during the beginning of the pandemic. Certain tech stock like Apple have a very steady trading volume, while others like Nvidia have a very fluctuating trade volume. Also noticeable are the very high trade volumes during the massive dip of Neflix and Meta in 2022, while the trading volume pattern of amazon stayed relatively consistent throughout their dip.
### 3. Volatility Analysis of Crypto vs. Financial Markets
![image alt](https://github.com/LewisSchmidtke/DataSciencePortfolio/blob/main/Plots/Volatility_Tech+Crypto.png?raw=true)</br>
I wanted to compare the volatility of crypto currencies compared to standard financial markets like the Nasdaq 100 and the S&P 500 to check investment rist and overall safety when investing. It is very obvious that the financial markets have very low volatility compared to crypto currencies, with even the outliers being only a little higher than the median volatility of crypto currencies
### 4. Market Composition of the S&P500 with high Teck-stock Dependency
![image alt](https://github.com/LewisSchmidtke/DataScience_ML_Portfolio/blob/main/Plots/S&P500_Market_Composition.png?raw=true)
I also analyzed the composition of the S&P 500 and how it has evolved in recent years. Due to the AI boom, the top eight tech stocks have risen significantly in price and now account for a substantial portion of the S&P 500's value. Given that the index includes 500 companies, it can be risky for just eight of them to have such a large influence, as this makes the S&P 500 more susceptible to volatility if those top stocks underperform.
### 5. Risk-Return Analysis of Tech Stocks and other Assets
![image alt](https://github.com/LewisSchmidtke/DataScience_ML_Portfolio/blob/main/Plots/risk-return_comparison.png?raw=true)
Lastly (for now), I tried analyzing the correlation between stock volatility (risk) and the potential profit (return) over 6 month periods. For this I divided each year into two halfs and separately calculated risk and return for each asset available (stocks and minerals). High profit fluctuation is visible for the Tesla stock which yielded the highest, as well as the lowest ROI, while also being extremely volatile. Due to the AI-Boom Nvidis has been constantly delivering high ROIs with a relatively low risk.
### 6.First Machine Learning Use-Case: Time Series Forecasting of the Nasdaq 100
![image alt](https://github.com/LewisSchmidtke/DataSciencePortfolio/blob/main/Plots/real_vs_predicted_price_nasdaq100.png?raw=true)
This approach utilizes a multivariate approach to predict the Nasdaq100 Price. With the use of an LSTM model, which takes the prize and trade volume of previous days as an input, the following day is predicted. The training is conducted with overlapping sliding windows of size 10, meaning the price and trade volume of the Nasdaq100 of the previous 10 days is considered when predicting the price for the following day. The graphic shows the forecasting capabilities of the LSTM model when applied to unseen test data, meaning these data points are explicitly set aside for testing purposes and have not been utilized during the training process.


## Global Cybersecurity Threats Plots
The following plots were created to visualize different aspects of the Cybersecurity data and to be serve as insights for future business decisions.
The data consists of more than 3000 entries in 10 columns.
### 1. Cyberattack Trends:
![image alt](https://github.com/LewisSchmidtke/DataSciencePortfolio/blob/main/Plots/Cyberattack_Trends.png?raw=true)
### 2. Industry Cyberattack Impact:
![image alt](https://github.com/LewisSchmidtke/DataSciencePortfolio/blob/main/Plots/Cyberattack_Impacts_Users&Finance.png?raw=true)
### 3. Attack Type / Target Industry Heatmap
![image alt](https://github.com/LewisSchmidtke/DataSciencePortfolio/blob/main/Plots/FinancialLoss_HeatMap.png?raw=true)
