import pandas as pd

df = pd.read_csv("data.csv")

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing Score with 0
df['Score'] = df['Score'].fillna(0)

print("\nCleaned Data:\n", df)