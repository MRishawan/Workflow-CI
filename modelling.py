import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import xgboost as xgb 

def train_and_log_model():
    # ==========================================
    # 1. Konfigurasi MLflow Tracking
    # ==========================================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Housing_Price")

    # ==========================================
    # 2. Memuat Dataset Preprocessed
    # ==========================================
    print("Memuat dataset hasil preprocessing...")
    try:
        train_df = pd.read_csv('housing_train_preprocessing.csv')
        test_df = pd.read_csv('housing_test_preprocessing.csv')
    except FileNotFoundError:
        print("Error: File CSV tidak ditemukan. Pastikan letak file sudah benar.")
        return

    # Memisahkan Fitur (X) dan Target (y)
    X_train = train_df.drop("median_house_value", axis=1)
    y_train = train_df["median_house_value"]
    
    X_test = test_df.drop("median_house_value", axis=1)
    y_test = test_df["median_house_value"]

    X_train.columns = X_train.columns.str.replace(r'[\[\]<]', '_', regex=True)
    X_test.columns = X_test.columns.str.replace(r'[\[\]<]', '_', regex=True)

    # ==========================================
    # 3. Memulai Sesi MLflow Run & Training
    # ==========================================
    with mlflow.start_run(run_name="XGBoost_Baseline_No_Tuning"):
        print("Memulai proses training XGBoost Baseline...")
        
        # Inisialisasi model baseline XGBoost
        # Menentukan beberapa parameter dasar agar tercatat di log
        baseline_params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': -1
        }
        
        xgb_model = xgb.XGBRegressor(**baseline_params)
        
        # Fitting model ke data latih
        xgb_model.fit(X_train, y_train)

        # ==========================================
        # 4. Evaluasi Model
        # ==========================================
        print("Melakukan prediksi dan kalkulasi metrik...")
        
        # Evaluasi Data Latih
        train_preds = xgb_model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_r2 = r2_score(y_train, train_preds)
        
        # Evaluasi Data Uji
        test_preds = xgb_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)

        # ==========================================
        # 5. Manual Logging ke MLflow
        # ==========================================
        print("Menyimpan log dan artefak ke MLflow...")
        
        # Mencatat parameter
        mlflow.log_params(baseline_params)
        
        # Mencatat metrik latih dan uji
        mlflow.log_metrics({
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        })
        
        # Mencatat file model dengan nama folder spesifik
        mlflow.sklearn.log_model(
            sk_model=xgb_model,
            artifact_path="XGBoost Model"
        )

        print("\n--- Hasil Evaluasi Test Set (XGBoost Baseline) ---")
        print(f"Test RMSE : ${test_rmse:,.2f}")
        print(f"Test R2   : {test_r2:.4f}")
        print("\nSelesai! Silakan cek MLflow UI Anda.")

if __name__ == "__main__":
    train_and_log_model()