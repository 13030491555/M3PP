import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False

x = pd.read_excel('data.xlsx')
x = x.dropna()
nunique = x.nunique()
cols_to_drop = nunique[nunique == 1].index
x = x.drop(cols_to_drop, axis=1)
x = x[x.iloc[:, -1] >= 10]
data = x.iloc[:, :51].to_numpy()
data = data - data.min(axis=0)
data = data / (data.max(axis=0) - data.min(axis=0) + 1e-8)

target_names = ['Tensile Strength (MPa)', 'Elongation (%)', 'Yield Strength (MPa)']

feature_sets = [
    (slice(0, 30), "C-P Mode (Columns 1~30)"),
    (slice(0, 47), "C-P-P Mode (Columns 1~47)"),
    (slice(0, 51), "C-P-M-P Mode (Columns 1~51)")
]

fig, axes = plt.subplots(9, 1, figsize=(7, 36))

if axes.ndim == 1:
    axes = axes.reshape(-1, 1)

for t_idx, target_name in enumerate(target_names):
    y = x[target_name].to_numpy()
    for f_idx, (feat_slice, feat_title) in enumerate(feature_sets):
        row_idx = t_idx * 3 + f_idx
        X = data[:, feat_slice]
        X_scaled = X / np.maximum(X.max(axis=0), 1e-8)
        X_scaled_cnn = X_scaled.reshape(-1, X_scaled.shape[1], 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_cnn, y, test_size=0.1, random_state=42
        )

        model_cnn = Sequential()
        model_cnn.add(Input(shape=(X.shape[1], 1)))
        model_cnn.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation='leaky_relu'))
        model_cnn.add(Dropout(0.1))
        model_cnn.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation='leaky_relu'))
        model_cnn.add(Dropout(0.1))
        model_cnn.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation='leaky_relu'))
        model_cnn.add(Conv1D(8, kernel_size=3, strides=1, padding='same', activation='leaky_relu'))
        model_cnn.add(Conv1D(4, kernel_size=3, strides=1, padding='same', activation='leaky_relu'))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(64, activation='leaky_relu'))
        model_cnn.add(Dense(1, activation='linear'))

        early_stop = EarlyStopping(
            monitor='val_loss', min_delta=0.01, restore_best_weights=True, patience=20
        )

        def lr_decay(epoch):
            initial_lr = 0.001
            decay_rate = 0.1
            decay_steps = 5
            new_lr = initial_lr * (decay_rate ** (epoch // decay_steps))
            return new_lr

        lr_decay_callback = LearningRateScheduler(lr_decay)

        model_cnn.compile(optimizer=Adam(learning_rate=0.005), loss='mean_squared_error')
        history_cnn = model_cnn.fit(
            X_train, y_train,
            batch_size=64,
            epochs=200,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, lr_decay_callback],
            verbose=0
        )

        y_pred_cnn = model_cnn.predict(X_test).flatten()
        p_r_cnn, _ = pearsonr(y_pred_cnn, y_test)
        var_cnn = explained_variance_score(y_test, y_pred_cnn)
        mse_cnn = mean_squared_error(y_test, y_pred_cnn)
        rmse_cnn = np.sqrt(mse_cnn)

        ax_cnn = axes[row_idx, 0]
        sns.regplot(x=y_pred_cnn, y=y_test, ax=ax_cnn, scatter_kws={'s': 25, 'alpha': 0.7})
        ax_cnn.set_xlabel('Predicted Value')
        ax_cnn.set_ylabel('True Value')
        ax_cnn.set_title(f"{feat_title} | {target_name}\nCNN")
        ax_cnn.text(
            0.05, 0.95,
            f"Pearson Correlation={p_r_cnn:.3f}\nRMSE={rmse_cnn:.2f}",
            ha='left', va='top',
            transform=ax_cnn.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

plt.tight_layout()
plt.show()
