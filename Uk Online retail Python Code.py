# ===============================
# UK Online Retail Starter Notebook
# ===============================

# step 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#step 1: Load Data
df = pd.read_csv(r"C:/Users/madem/OneDrive/Desktop/Data Analytics/data.csv", encoding = 'ISO-8859-1')
print("Dataset Loaded")
print(df.head())
print(df.info())
print(df.describe())

#step 2: Data Cleaning
#Removing Missing CustomerID

missing_values = df.isnull().sum()
print(missing_values)

df = df.dropna(subset=['CustomerID'])

missing_values = df.isnull().sum()
print(missing_values)

#Remove negative/Zero Quantity and UnitPrice
df = df[(df['Quantity'] > 0) &(df['UnitPrice'] > 0)]

#Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day


#create Revenue Column
df['Revenue'] = df['Quantity'] * df['UnitPrice']

 
print("\nCleaned Dataset Info:")
print(df.info())


# Step 3: Descriptive Analysis
# Total Revenue, Orders, Average Order Value
total_revenue = df['Revenue'].sum()
total_orders = df['InvoiceNo'].nunique()
avg_order_value = total_revenue / total_orders

print(f"\nTotal Revenue: £{total_revenue:,.2f}")
print(f"Total Orders: {total_orders}")
print(f"Average Order Value: £{avg_order_value:,.2f}")

# Top 10 Products by Revenue
top_products = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Revenue:")
print(top_products)



# Monthly Revenue Trend
monthly_sales = df.groupby(['Year','Month'])['Revenue'].sum().reset_index()
monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str)

plt.figure(figsize=(12,6))
sns.lineplot(data=monthly_sales, x='YearMonth', y='Revenue', marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Revenue Trend")
plt.ylabel("Revenue (£)")
plt.xlabel("Month")
plt.show()


# Revenue by Country
country_sales = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=country_sales.index[:10], y=country_sales.values[:10])
plt.xticks(rotation=45)


# Step 4: Customer Analysis
# New vs Repeat Customers
customer_orders = df.groupby('CustomerID')['InvoiceNo'].nunique()
repeat_customers = (customer_orders > 1).sum()
new_customers = (customer_orders == 1).sum()
print(f"\nNew Customers: {new_customers}")
print(f"Repeat Customers: {repeat_customers}")

# Pie Chart
plt.figure(figsize=(6,6))
plt.pie([new_customers, repeat_customers], labels=['New', 'Repeat'], autopct='%1.1f%%', colors=['skyblue','orange'])
plt.title("Customer Distribution: New vs Repeat")
plt.show()



# Step 5: Revenue Forecasting
from prophet import Prophet

# Prepare data
monthly_revenue = df.groupby('InvoiceDate')['Revenue'].sum().reset_index()
monthly_revenue.rename(columns={'InvoiceDate':'ds', 'Revenue':'y'}, inplace=True)

# Initialize and fit model
model = Prophet()
model.fit(monthly_revenue)

# Forecast next 3 months (90 days)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
fig.show()
plt.xlabel("Country")
plt.show()


# Pie Chart
plt.figure(figsize=(6,6))
plt.pie([new_customers, repeat_customers], labels=['New', 'Repeat'], autopct='%1.1f%%', colors=['skyblue','orange'])
plt.title("Customer Distribution: New vs Repeat")
plt.show()





# Step 6: RFM Analysis
import datetime as dt

# Snapshot date = last transaction + 1 day
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Aggregate RFM metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'Revenue': 'sum'                                         # Monetary
})

rfm.rename(columns={'InvoiceDate':'Recency', 'InvoiceNo':'Frequency', 'Revenue':'Monetary'}, inplace=True)

# View top rows
print(rfm.head())

# Optional: Assign RFM scores (1-5)
r_labels = range(5, 0, -1)  # 5 = most recent
f_labels = range(1, 6)      # 5 = most frequent
m_labels = range(1, 6)      # 5 = highest spender

rfm['R_Score'] = pd.cut(rfm['Recency'], 5, labels=r_labels)
rfm['F_Score'] = pd.cut(rfm['Frequency'], 5, labels=f_labels)
rfm['M_Score'] = pd.cut(rfm['Monetary'], 5, labels=m_labels)

# Combine RFM score
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Example: View top 10 high-value customers
top_customers = rfm.sort_values('RFM_Score', ascending=False).head(10)
print(top_customers)








