from file_utils import load_data

df = load_data()

print("First rows:\n")
print(df.head())

print("Structural information:\n")
print(df.info())

print("Descriptive statistics:\n")
print(df.describe(include="all"))

print("Missing values:\n")
print(df.isnull().sum())

print("Duplicate rows:\n")
print("Number of duplicates:", df.duplicated().sum())

print("Sorting by money:\n")
sorted_df = df.sort_values(by="money", ascending=False)
print(sorted_df.head(10))
