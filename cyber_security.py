import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize kaggle api call
# api = KaggleApi()
# api.authenticate()
# api.dataset_download_files('atharvasoundankar/global-cybersecurity-threats-2015-2024', path='data/', unzip=True)


# Extract data
df = pd.read_csv('data/Global_Cybersecurity_Threats_2015-2024.csv')

categories = list(df.columns)

# Extract data for the whole year
nr_of_attacks_per_year = df["Year"].value_counts().reset_index().sort_values(by='Year', ascending=True)
money_lost_per_year = df.groupby('Year')['Financial Loss (in Million $)'].sum().reset_index()
affected_users_per_year = df.groupby('Year')['Number of Affected Users'].sum().reset_index()
time_working_on_resolution_per_year = df.groupby('Year')['Incident Resolution Time (in Hours)'].sum().reset_index()
# Merge to one df
data_per_year = pd.merge(nr_of_attacks_per_year, money_lost_per_year, on='Year', how='outer')
data_per_year = pd.merge(data_per_year, affected_users_per_year, on='Year', how='outer')
data_per_year = pd.merge(data_per_year, time_working_on_resolution_per_year, on='Year')
data_per_year["Financial Loss (in 100 Million $)"] = data_per_year["Financial Loss (in Million $)"] / 100
data_per_year["Number of Affected Users (in Million)"] = data_per_year["Number of Affected Users"] / 1000000

# 1. ----------------- General Trends per year: Nr of Attacks per year per Type -----------------

# Extracting attacks per year per attack type and applying 3 year rolling average to uncover general trends
attack_trends = df.groupby(["Year", "Attack Type"]).size().reset_index(name="Number of Attacks")
attack_trends["Smoothed Attacks"] = attack_trends.groupby("Attack Type")["Number of Attacks"].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Customizing figure aesthetics
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 6))
plt.title("Cyberattack Trends: 3-Year Rolling Average", fontsize=22, fontweight="bold", pad=20)
plt.xlabel("Year", fontsize=14, labelpad=10)
plt.ylabel("Number of Attacks", fontsize=14, labelpad=10)
plt.legend(title="Attack Type", bbox_to_anchor=(1, 1), loc="upper left")

# Plotting the data
sns.lineplot(data=attack_trends, x="Year", y="Smoothed Attacks", hue="Attack Type",
             palette="deep", alpha=.7, linewidth=2, marker="o")
plt.tight_layout()
plt.show()
plt.close()



# 2. ----------------- Correlation between Attack type, Financial Loss and Targeted Industry -----------------
# Transform data to be used in Heatmap-style plot
heatmap_data = df.pivot_table(
    index="Attack Type",
    columns="Target Industry",
    values="Financial Loss (in Million $)",
    aggfunc="mean"
)

# Define figure size and create it
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(
    heatmap_data,
    cmap="Reds",  # Red color scale for financial loss
    annot=True,  # Show values
    fmt=".1f",  # Format values with one decimal
    linewidths=0.5,
    linecolor="gray",
)
# Customize Aesthetics
plt.title("Financial Loss by Industry & Attack Type (Million USD)", fontsize=18, fontweight="bold", pad=20)
plt.xlabel("Targeted Industry", fontsize=14, labelpad=10)
plt.ylabel("Attack Type", fontsize=14, labelpad=10)
plt.xticks(rotation=45)
colorbar = heatmap.collections[0].colorbar
colorbar.set_label('Financial Loss (in Million $)')
plt.tight_layout()
plt.show()


# 3. ----------------- Financial Loss vs. Number of Affected Users by Targeted Industry -----------------
df["Number of Affected Users in 10000"] = df["Number of Affected Users"] / 10000
df_melted = df.melt(id_vars=["Target Industry"], value_vars=["Financial Loss (in Million $)", "Number of Affected Users in 10000"],
                    var_name="Metric", value_name="Value")

sns.set_theme(style="ticks", palette="muted")
plt.figure(figsize=(15, 6))

# Boxplot with hue set to the 'Metric' column to differentiate the two metrics
sns.boxplot(
    data=df_melted,
    x="Target Industry",
    y="Value",
    hue="Metric",
    palette=["#1f77b4", "#ff7f0e"]
)
plt.title("Industry Cyberattack Impact: Financial Loss vs. Users Impacted", fontsize=16, fontweight='bold')
plt.legend(fontsize=12, bbox_to_anchor=(1, 1.12))
plt.xlabel("Target Industry", fontsize=14, labelpad=14)
plt.ylabel("")
sns.despine(offset=20, trim=True)
plt.tight_layout()
plt.show()

