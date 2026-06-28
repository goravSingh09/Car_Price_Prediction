import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv("car data.csv")

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# =========================
# STEP 2: Encode Categorical Data
# =========================
df.replace({
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
    'Selling_type': {'Dealer': 0, 'Individual': 1},
    'Transmission': {'Manual': 0, 'Automatic': 1}
}, inplace=True)

print("\nEncoded Dataset:")
print(df.head())

# =========================
# STEP 3: Features and Target
# =========================
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# =========================
# STEP 4: Split Dataset
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# =========================
# STEP 5: Train Model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# =========================
# STEP 6: Check Accuracy
# =========================
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print("\nModel Accuracy (R2 Score):")
print(round(score, 2))

# =========================
# STEP 7: User Input Prediction
# =========================
print("\n===== Car Price Prediction =====")

year = int(input("Enter Car Year: "))
present_price = float(input("Enter Present Price (in lakhs): "))
driven_kms = int(input("Enter Kilometers Driven: "))
fuel_type = int(input("Enter Fuel Type (Petrol=0, Diesel=1, CNG=2): "))
selling_type = int(input("Enter Selling Type (Dealer=0, Individual=1): "))
transmission = int(input("Enter Transmission (Manual=0, Automatic=1): "))
owner = int(input("Enter Number of Previous Owners: "))

new_car = pd.DataFrame([[
    year,
    present_price,
    driven_kms,
    fuel_type,
    selling_type,
    transmission,
    owner
]], columns=X.columns)

predicted_price = model.predict(new_car)

print("\nPredicted Selling Price:")
print(round(predicted_price[0], 2), "Lakhs")