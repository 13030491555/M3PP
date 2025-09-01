import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.1, random_state=42
        )

        model_rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        p_r_rf, _ = pearsonr(y_pred_rf, y_test)
        var_rf = explained_variance_score(y_test, y_pred_rf)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mse_rf)

        ax_rf = axes[row_idx, 0]
        sns.regplot(x=y_pred_rf, y=y_test, ax=ax_rf, scatter_kws={'s': 25, 'alpha': 0.7})
        ax_rf.set_xlabel('Predicted Value')
        ax_rf.set_ylabel('True Value')
        ax_rf.set_title(f"{feat_title} | {target_name}\nRandom Forest")
        ax_rf.text(
            0.05, 0.95,
            f"Pearson Correlation={p_r_rf:.3f}\nRMSE={rmse_rf:.2f}",
            ha='left', va='top',
            transform=ax_rf.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

plt.tight_layout()
plt.show()
