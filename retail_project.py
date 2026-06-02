import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("superstore.csv", encoding="latin1")

print("Dataset Shape:", df.shape)

# -------------------
# Sales by Category
# -------------------
sales_category = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8,5))
sales_category.plot(kind="bar")

plt.title("Retail Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("retail_sales_category.png")
plt.close()

# -------------------
# Sales by Region
# -------------------
sales_region = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(8,5))
sales_region.plot(kind="bar")

plt.title("Retail Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("retail_sales_region.png")
plt.close()

# -------------------
# Prediction Model
# -------------------
X = df[["Quantity", "Discount"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print("Model Accuracy:", score)

print("Retail Project Completed Successfully")