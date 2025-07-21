CSV_FILE = "data.csv"

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time


class GripperSVMController:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.metrics = {}
        self.feature_names = [
            'current_time', 'current_bending', 'current_pwm',
            'bending_rate', 'pwm_efficiency', 'rise_slope',
            'stability_factor', 'acceleration_phase'
        ]

    def extract_features(self, ts):
        if len(ts) < 3:
            return np.zeros(8)
        t = [p[0] for p in ts]
        pwm = [p[1] for p in ts]
        bend = [p[2] for p in ts]
        cur_t, cur_pwm, cur_bend = t[-1], pwm[-1], bend[-1]
        rate = (bend[-1] - bend[-5]) / (t[-1] - t[-5]) if len(bend) >= 5 else 0
        eff = cur_bend / cur_pwm if cur_pwm > 0 else 0
        slope = np.polyfit(t[-10:], bend[-10:], 1)[0] if len(bend) >= 10 else 0
        stab = 1 / (1 + np.var(bend[-5:])) if len(bend) >= 5 else 0.5
        accel = cur_bend / cur_t if cur_t > 0 else 0
        return np.array([cur_t, cur_bend, cur_pwm, rate, eff, slope, stab, accel])

    def prepare_data(self, csv_path):
        df = pd.read_csv(csv_path)
        X, y = [], []
        for trial in df["trial_number"].unique():
            d = df[df["trial_number"] == trial].sort_values("time_seconds")
            ts = d[["time_seconds", "pump_speed_pwm", "bending_value"]].values.tolist()
            if len(ts) < 10:
                continue
            idxs = [int(len(ts) * r) for r in [0.3, 0.5, 0.7, 0.9]] + [len(ts) - 1]
            for i in idxs:
                if i < len(ts):
                    feat = self.extract_features(ts[:i + 1])
                    target = 1.0 if i / len(ts) > 0.8 else 0.0
                    X.append(feat)
                    y.append(target)
        return np.array(X), np.array(y)

    def train(self, csv_path):
        X, y = self.prepare_data(csv_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1],
            'epsilon': [0.01, 0.1],
            'kernel': ['rbf']
        }

        search = GridSearchCV(SVR(), param_grid, cv=5,
                              scoring='neg_mean_squared_error', n_jobs=-1)
        start = time.time()
        search.fit(X_train_scaled, y_train)
        duration = time.time() - start

        self.model = search.best_estimator_
        self.is_trained = True

        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        self.metrics = {
            "r2": r2,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "training_time": duration,
            "best_params": search.best_params_
        }
        return self.metrics

    def predict(self, ts):
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        feat = self.extract_features(ts)
        scaled = self.scaler.transform([feat])
        return float(np.clip(self.model.predict(scaled)[0], 0, 1))

    def save(self, prefix):
        if self.is_trained:
            joblib.dump(self.model, f"{prefix}_model.pkl")
            joblib.dump(self.scaler, f"{prefix}_scaler.pkl")
            joblib.dump(self.metrics, f"{prefix}_metrics.pkl")

    def load(self, prefix):
        self.model = joblib.load(f"{prefix}_model.pkl")
        self.scaler = joblib.load(f"{prefix}_scaler.pkl")
        try:
            self.metrics = joblib.load(f"{prefix}_svm_metrics.pkl")
        except:
            self.metrics = {}
        self.is_trained = True


if __name__ == "__main__":
    controller = GripperSVMController()
    metrics = controller.train(CSV_FILE)
    controller.save("gripper_svm")
    print(f"R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")


