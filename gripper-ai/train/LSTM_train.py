CSV_FILE = "data.csv"

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
import time

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

warnings.filterwarnings('ignore')

np.random.seed(42)
if DEEP_LEARNING_AVAILABLE:
    tf.random.set_seed(42)


class GripperAIController:
    def __init__(self):
        self.model = None
        self.lstm_model = None
        self.scaler = None
        self.is_trained = False
        self.is_lstm_trained = False
        self.sequence_length = 76
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
                    'time_series': time_series,
                    'final_pwm': time_series[-1][1]
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

    def prepare_sequence_data(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        sequences = []
        targets = []

        for trial_num in df['trial_number'].unique():
            trial_data = df[df['trial_number'] == trial_num].sort_values('time_seconds')
            if len(trial_data) < self.sequence_length:
                continue

            raw_sequence = [
                [row['time_seconds'], row['pump_speed_pwm'], row['bending_value']]
                for _, row in trial_data.iterrows()
            ]

            for i in range(self.sequence_length, len(raw_sequence)):
                sequence_window = raw_sequence[i-self.sequence_length:i]
                sequence_features = [
                    [p[0]/15.0, (p[1]-80)/170.0, p[2]/60.0] for p in sequence_window
                ]
                progress_ratio = i / len(raw_sequence)
                target = 1.0 if progress_ratio > 0.8 else 0.0
                sequences.append(sequence_features)
                targets.append(target)

        return np.array(sequences), np.array(targets)

    def train_model(self, csv_file_path):
        X, y = self.prepare_training_data(csv_file_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.is_trained = True
        return mse, r2

    def train_lstm(self, csv_file_path):
        if not DEEP_LEARNING_AVAILABLE:
            return None

        start_time = time.time()
        X_seq, y_seq = self.prepare_sequence_data(csv_file_path)
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        model = Sequential([
            LSTM(32, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
        ]

        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=50, batch_size=32, callbacks=callbacks, verbose=1)

        y_pred = model.predict(X_test, verbose=0).flatten()

        self.lstm_model = model
        self.is_lstm_trained = True

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mse, r2

    def predict_stop_probability(self, current_time_series, use_lstm=False):
        if use_lstm and self.is_lstm_trained:
            return self.predict_lstm_stop_probability(current_time_series)

        if not self.is_trained:
            raise ValueError("Random Forest model not trained")

        features = self.extract_features(current_time_series)
        features_scaled = self.scaler.transform([features])
        return np.clip(self.model.predict(features_scaled)[0], 0, 1)

    def predict_lstm_stop_probability(self, current_time_series):
        if not self.is_lstm_trained:
            raise ValueError("LSTM model not trained")

        if len(current_time_series) < self.sequence_length:
            return 0.0

        sequence_features = [
            [p[0]/15.0, (p[1]-80)/170.0, p[2]/60.0] for p in current_time_series[-self.sequence_length:]
        ]

        sequence_array = np.array([sequence_features])
        return np.clip(self.lstm_model.predict(sequence_array, verbose=0)[0][0], 0, 1)

    def save_model(self, filepath_prefix):
        if self.is_trained:
            joblib.dump(self.model, f"{filepath_prefix}_model.pkl")
            joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")
        if self.is_lstm_trained:
            self.lstm_model.save(f"{filepath_prefix}_lstm_model.h5")

    def load_model(self, filepath_prefix):
        try:
            self.model = joblib.load(f"{filepath_prefix}_model.pkl")
            self.scaler = joblib.load(f"{filepath_prefix}_scaler.pkl")
            self.is_trained = True
        except FileNotFoundError:
            pass

        if DEEP_LEARNING_AVAILABLE:
            try:
                from tensorflow.keras.models import load_model
                self.lstm_model = load_model(f"{filepath_prefix}_lstm_model.h5")
                self.is_lstm_trained = True
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    controller = GripperAIController()
    try:
        mse_rf, r2_rf = controller.train_model(CSV_FILE)
        if DEEP_LEARNING_AVAILABLE:
            mse_lstm, r2_lstm = controller.train_lstm(CSV_FILE)
        controller.save_model("gripper_model")

    except Exception as e:
        print(f"Error: {e}")
