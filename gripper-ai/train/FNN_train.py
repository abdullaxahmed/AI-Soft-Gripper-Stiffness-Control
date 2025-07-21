CSV_FILE = "data.csv"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
import time

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

warnings.filterwarnings('ignore')

np.random.seed(42)
if DEEP_LEARNING_AVAILABLE:
    tf.random.set_seed(42)


class EnhancedGripperAIController:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = None
        self.performance_metrics = {}
        self.feature_names = [
            'current_time', 'current_bending', 'current_pwm',
            'bending_rate', 'pwm_efficiency', 'rise_slope',
            'stability_factor', 'acceleration_phase'
        ]

    def extract_features(self, time_series_data):
        if len(time_series_data) < 3:
            return np.zeros(8)

        times = [point[0] for point in time_series_data]
        pwms = [point[1] for point in time_series_data]
        bendings = [point[2] for point in time_series_data]

        current_time = times[-1]
        current_bending = bendings[-1]
        current_pwm = pwms[-1]

        if len(bendings) >= 5:
            recent_bendings = bendings[-5:]
            recent_times = times[-5:]
            bending_rate = (recent_bendings[-1] - recent_bendings[0]) / (recent_times[-1] - recent_times[0])
        else:
            bending_rate = 0

        pwm_efficiency = current_bending / current_pwm if current_pwm > 0 else 0

        if len(bendings) >= 10:
            x = np.array(times[-10:])
            y = np.array(bendings[-10:])
            rise_slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
        else:
            rise_slope = 0

        if len(bendings) >= 5:
            stability_factor = 1.0 / (1.0 + np.var(bendings[-5:]))
        else:
            stability_factor = 0.5

        acceleration_phase = current_bending / current_time if current_time > 0 else 0

        return np.array([
            current_time, current_bending, current_pwm,
            bending_rate, pwm_efficiency, rise_slope,
            stability_factor, acceleration_phase
        ])

    def prepare_training_data(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        trials = []
        for trial_num in df['trial_number'].unique():
            trial_data = df[df['trial_number'] == trial_num].sort_values('time_seconds')
            time_series = [
                [row['time_seconds'], row['pump_speed_pwm'], row['bending_value']]
                for _, row in trial_data.iterrows()
            ]
            if len(time_series) > 10:
                trials.append({
                    'trial_number': trial_num,
                    'time_series': time_series
                })

        X_features = []
        y_targets = []

        for trial in trials:
            time_series = trial['time_series']
            sample_points = [
                int(len(time_series) * 0.3),
                int(len(time_series) * 0.5),
                int(len(time_series) * 0.7),
                int(len(time_series) * 0.9),
                len(time_series) - 1
            ]
            for point_idx in sample_points:
                if point_idx < len(time_series):
                    current_series = time_series[:point_idx + 1]
                    features = self.extract_features(current_series)
                    progress_ratio = point_idx / len(time_series)
                    should_stop_soon = 1.0 if progress_ratio > 0.8 else 0.0
                    X_features.append(features)
                    y_targets.append(should_stop_soon)

        return np.array(X_features), np.array(y_targets)

    def train_fnn(self, csv_file_path):
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TensorFlow not available")

        start_time = time.time()
        X, y = self.prepare_training_data(csv_file_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['mae', 'mse']
        )

        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
        ]

        self.training_history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        y_pred = self.model.predict(X_test_scaled, verbose=0).flatten()

        training_time = time.time() - start_time
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.performance_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time,
            'epochs_trained': len(self.training_history.history['loss']),
            'final_loss': self.training_history.history['loss'][-1],
            'final_val_loss': self.training_history.history['val_loss'][-1]
        }

        self.is_trained = True
        return mse, r2

    def predict_stop_probability(self, current_time_series):
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        features = self.extract_features(current_time_series)
        features_scaled = self.scaler.transform([features])
        stop_probability = self.model.predict(features_scaled, verbose=0)[0][0]
        return np.clip(stop_probability, 0, 1)

    def save_model(self, filepath_prefix):
        if self.is_trained:
            self.model.save(f"{filepath_prefix}_fnn_model.h5")
            joblib.dump(self.scaler, f"{filepath_prefix}_fnn_scaler.pkl")
            joblib.dump(self.performance_metrics, f"{filepath_prefix}_fnn_metrics.pkl")

    def load_model(self, filepath_prefix):
        if DEEP_LEARNING_AVAILABLE:
            self.model = tf.keras.models.load_model(f"{filepath_prefix}_fnn_model.h5")
            self.scaler = joblib.load(f"{filepath_prefix}_fnn_scaler.pkl")
            self.performance_metrics = joblib.load(f"{filepath_prefix}_fnn_metrics.pkl")
            self.is_trained = True


if __name__ == "__main__":
    controller = EnhancedGripperAIController()
    try:
        mse, r2 = controller.train_fnn(CSV_FILE)
        controller.save_model("enhanced_gripper")

    except Exception as e:
        print(f"Error: {e}")
print(f"R²: {r2:.4f}, MSE: {mse:.4f}")