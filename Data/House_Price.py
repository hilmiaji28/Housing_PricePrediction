## Cari fitur numerik yang paling penting untuk dianalisis

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

# load data
df = pd.read_csv("train.csv")

# ambil hanya fitur numerik
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# target & feature
X = numeric_df.drop("SalePrice", axis=1)
y = numeric_df["SalePrice"]

# handling missing value
imputer = SimpleImputer(strategy='median')

X_imputed = imputer.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=0.2,
    random_state=42
)

# model XGBoost
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# train model
model.fit(X_train, y_train)

# prediksi
y_pred = model.predict(X_test)

# evaluasi
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# feature importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

# tampilkan top 4
print("\nTop 4 Numerical Feature Importance:")
print(feature_importance.head(4))

# visualisasi
top_n = 4

plt.figure(figsize=(12,8))

plt.barh(
    feature_importance['Feature'].head(top_n)[::-1],
    feature_importance['Importance'].head(top_n)[::-1]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 4 Numerical Feature Importance (XGBoost)")

plt.tight_layout()
plt.show()

#keterangan fitur (berdasarkan fitur numerik yang paling penting)
# OverallQual : Kualitas keseluruhan rumah (1)
# GarageCars : Jumlah garasi (2)
# FullBath : Jumlah kamar mandi (3)
# GrLivArea : Luas area tinggal (4)
# ambil 3 paling penting yaitu overallqual, garagecars, dan grlivarea