import pandas as pd
import numpy as np

np.random.seed(42)

def generate_housing_data(n_samples=1000, drift=False):
    """Generate dummy housing data berdasarkan fitur numerik Ames Housing"""

    if drift:
        # Simulasi data drift
        OverallQual = np.random.randint(5, 11, n_samples) #kualitas rumah yang lebih baik dari sebelumnya
        GarageCars = np.random.randint(1, 5, n_samples) #jumlah garasi yang lebih banyak dari sebelumnya
        GrLiveArea = np.random.normal(2200, 500, n_samples) #luas area tinggal yang lebih besar dari sebelumnya

    else:
        OverallQual = np.random.randint(1, 10, n_samples) 
        GarageCars = np.random.randint(0, 4, n_samples)
        GrLiveArea = np.random.normal(1500, 400, n_samples)

    # Formula harga rumah + noise
    SalePrice = (
        OverallQual * 50000 +
        GarageCars * 20000 +
        GrLiveArea * 120 +
        np.random.normal(0, 30000, n_samples)
    )

    df = pd.DataFrame({
        'OverallQual': OverallQual,
        'GarageCars': GarageCars,
        'GrLiveArea': GrLiveArea,
        'SalePrice': SalePrice
    })

    return df


if __name__ == "__main__":

    # Generate drift data
    drift_data = generate_housing_data(200, drift=True)
    drift_data.to_csv('drift_data.csv', index=False)
    print("✅ Drift data generated: drift_data.csv")