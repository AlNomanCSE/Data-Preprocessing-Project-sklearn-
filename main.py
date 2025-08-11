import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("sample_data.csv")
print(df.head())

# Step 1: Data Cleaning
# 1.1 Handle missing values and outliers
print("\nStep 1.1: Handling missing values and outliers")
# Handle outlier in Age (>100)
df.loc[df["Age"] > 100, "Age"] = df["Age"].median()  # Replace with median
# Fill missing Gender with mode
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])  # Mode for categorical
# Fill missing Income with median
df["Income"] = df["Income"].fillna(df["Income"].median())  # Median for numerical
print("After handling missing values and outliers (first 5 rows):\n", df.head())

# 1.2 Remove duplicates
print("\nStep 1.2: Removing duplicates")
print("Number of duplicate rows:", df.duplicated().sum())  # Expect 3 duplicates
df = df.drop_duplicates()  # Keep first occurrence
print("After removing duplicates (first 5 rows):\n", df.head())

# Step 2: Encoding
# 2.1 Label encoding for Gender
print("\nStep 2.1: Label Encoding for Gender")
df_label = df.copy()
le = LabelEncoder()
df_label["Gender_Encoded"] = le.fit_transform(df_label["Gender"])  # Male=1, Female=0
df_encoded = df_label.drop(columns=["Gender"])
print("After label encoding Gender (first 5 rows):\n", df_encoded.head())

# 2.2 One-hot encoding for Education and City
print("\nStep 2.2: One-Hot Encoding for Education and City")
ohe = OneHotEncoder(sparse_output=False)  # Dense output
oheData = ohe.fit_transform(df_encoded[["Education", "City"]])
featuresName = ohe.get_feature_names_out(["Education", "City"])
df_one_hot_encoded = pd.DataFrame(oheData, columns=featuresName)
print("One-hot encoded features (first 5 rows):\n", df_one_hot_encoded.head())

# Combine encoded features with the rest of the data
df_final = pd.concat(
    [df_encoded.drop(columns=["Education", "City"]), df_one_hot_encoded], axis=1
)
print("After combining all features (first 5 rows):\n", df_final.head())

# Step 3: Feature Scaling
print("\nStep 3: Feature Scaling")
df_scaled = df_final.copy()
# scaler = StandardScaler()  # Standardize Age and Income
# df_scaled[["Age", "Income"]] = scaler.fit_transform(df_scaled[["Age", "Income"]])
# Alternative: Use MinMaxScaler (uncomment to use)
scaler = MinMaxScaler()
df_scaled[["Age", "Income"]] = scaler.fit_transform(df_scaled[["Age", "Income"]])
print("After feature scaling (first 5 rows):\n", df_scaled.head())

# Step 4: Train-Test Split
print("\nStep 4: Train-Test Split")
# Define features (X) and target (y)
X = df_scaled.drop(columns=["Income"])  # Features: Age, Gender_Encoded, Education_*, City_*
y = df_scaled["Income"]  # Target: Income (scaled)
# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training features shape:", X_train.shape)  # Expect (38, ~9 features)
print("Testing features shape:", X_test.shape)   # Expect (9, ~9 features)
print("Training target shape:", y_train.shape)   # Expect (38,)
print("Testing target shape:", y_test.shape)     # Expect (9,)
print("\nTraining features (first 5 rows):\n", X_train.head())
print("\nTesting features (first 5 rows):\n", X_test.head())
print("\nTraining target (first 5 values):\n", y_train.head())
print("\nTesting target (first 5 values):\n", y_test.head())