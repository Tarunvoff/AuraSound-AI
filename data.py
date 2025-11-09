import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate sample dataset
np.random.seed(42)

cities = ['San Francisco', 'New York', 'Boston', 'Austin', 'Seattle', 
          'Los Angeles', 'Chicago', 'Denver', 'Miami', 'Atlanta', 
          'Portland', 'Dallas']
industries = ['FinTech', 'HealthTech', 'EdTech', 'E-commerce', 
              'AI/ML', 'SaaS', 'CleanTech', 'FoodTech']
stages = ['Seed', 'Series A', 'Series B', 'Series C', 'Series D']

data = {
    'City': np.random.choice(cities, 200),
    'Industry': np.random.choice(industries, 200),
    'Funding Stage': np.random.choice(stages, 200),
    'Investment Amount': np.random.randint(500000, 50000000, 200)
}

df = pd.DataFrame(data)

# Display basic information
print("First five records:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# 4. City-wise startup count
plt.figure(figsize=(8,5))
city_counts = df['City'].value_counts().head(10)
sns.barplot(x=city_counts.values, y=city_counts.index, palette='coolwarm')
plt.title("Top 10 Cities with Most Startups")
plt.xlabel("Number of Startups")
plt.ylabel("City")
plt.tight_layout()
plt.show()

# 5. Stage-wise funding histogram
plt.figure(figsize=(8,5))
stage_funding = df.groupby('Funding Stage')['Investment Amount'].sum()
sns.barplot(x=stage_funding.index, y=stage_funding.values, palette='viridis')
plt.title("Funding Distribution by Stage")
plt.xlabel("Funding Stage")
plt.ylabel("Total Investment Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Industry vs City heatmap
pivot = df.pivot_table(values='Investment Amount', 
                       index='Industry', 
                       columns='City', 
                       aggfunc='sum', 
                       fill_value=0)
# Select top 8 cities by total investment for better visualization
top_cities = df.groupby('City')['Investment Amount'].sum().nlargest(8).index
pivot_filtered = pivot[top_cities]

plt.figure(figsize=(10,6))
sns.heatmap(pivot_filtered, cmap='YlGnBu', fmt='.0f', linewidths=0.5)
plt.title("Industry vs City Heatmap of Investment Amount")
plt.xlabel("City")
plt.ylabel("Industry")
plt.tight_layout()
plt.show()