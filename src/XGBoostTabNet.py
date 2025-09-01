import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import os
import pickle
import joblib
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
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
    (slice(0, 30), "C-P (1~30)"),
    (slice(0, 47), "C-P-P (1~47)"),
    (slice(0, 51), "C-P-M-P (1~51)")
]

os.makedirs('saved_models', exist_ok=True)

all_models = {}
performance_summary = {}

fig, axes = plt.subplots(len(target_names) * len(feature_sets), 1,
                         figsize=(10, 5 * len(target_names) * len(feature_sets)))

plot_idx = 0
for t_idx, target_name in enumerate(target_names):
    y = x[target_name].to_numpy()

    all_models[target_name] = {}
    performance_summary[target_name] = {}

    target_best_pr = -1
    target_best_config = None

    for f_idx, (feat_slice, feat_title) in enumerate(feature_sets):
        X = data[:, feat_slice]
        X_scaled = X / np.maximum(X.max(axis=0), 1e-8)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.1, random_state=42
        )

        # ----------- XGBoost -----------
        model_xgb = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model_xgb.fit(X_train, y_train)
        y_pred_xgb_train = model_xgb.predict(X_train)
        y_pred_xgb_test = model_xgb.predict(X_test)

        # ----------- TabNet -----------
        model_tabnet = TabNetRegressor(
            n_d=32, n_a=32, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax',
            scheduler_params={"step_size": 20, "gamma": 0.8},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=0,
            seed=42
        )
        model_tabnet.fit(
            X_train=X_train, y_train=y_train.reshape(-1, 1),
            eval_set=[(X_test, y_test.reshape(-1, 1))],
            eval_metric=['rmse'],
            max_epochs=300,
            patience=30,
            batch_size=32,
            virtual_batch_size=16,
            num_workers=0,
            drop_last=False
        )
        y_pred_tabnet_train = model_tabnet.predict(X_train).flatten()
        y_pred_tabnet_test = model_tabnet.predict(X_test).flatten()

        # ----------- Stacking（XGBoost+TabNet） -----------
        stack_train = np.vstack([y_pred_xgb_train, y_pred_tabnet_train]).T
        stack_test = np.vstack([y_pred_xgb_test, y_pred_tabnet_test]).T
        stacker = LinearRegression()
        stacker.fit(stack_train, y_train)
        y_pred_stack = stacker.predict(stack_test)

        p_r_stack, _ = pearsonr(y_pred_stack, y_test)
        mse_stack = mean_squared_error(y_test, y_pred_stack)
        rmse_stack = np.sqrt(mse_stack)

        feature_key = feat_title.replace(" ", "_")
        all_models[target_name][feature_key] = {
            'xgboost': model_xgb,
            'tabnet': model_tabnet,
            'stacker': stacker,
            'feature_slice': feat_slice,
            'performance': {
                'pearson_r': p_r_stack,
                'rmse': rmse_stack,
                'mse': mse_stack
            }
        }

        performance_summary[target_name][feature_key] = {
            'pearson_r': p_r_stack,
            'rmse': rmse_stack
        }

        if p_r_stack > target_best_pr:
            target_best_pr = p_r_stack
            target_best_config = feature_key

        ax = axes[plot_idx]
        sns.regplot(x=y_pred_stack, y=y_test, ax=ax, scatter_kws={'s': 25, 'alpha': 0.7})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f"{target_name} | {feat_title}\nXGBoost+TabNet(Stacking)")
        ax.text(
            0.05, 0.95,
            f"Pearson r={p_r_stack:.3f}\nRMSE={rmse_stack:.2f}",
            ha='left', va='top',
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        plot_idx += 1
    performance_summary[target_name]['best_config'] = target_best_config

plt.tight_layout()
plt.savefig('xgboost_tabnet_performance.png', dpi=300, bbox_inches='tight')
plt.show()

for target_name, feature_models in all_models.items():
    target_dir = os.path.join('saved_models', target_name)
    os.makedirs(target_dir, exist_ok=True)

    for feature_key, model_dict in feature_models.items():
        feature_dir = os.path.join(target_dir, feature_key)
        os.makedirs(feature_dir, exist_ok=True)

        xgb_model_path = os.path.join(feature_dir, 'xgboost_model.json')
        model_dict['xgboost'].save_model(xgb_model_path)

        tabnet_model_path = os.path.join(feature_dir, 'tabnet_model.zip')
        model_dict['tabnet'].save_model(tabnet_model_path)

        stacker_model_path = os.path.join(feature_dir, 'stacker_model.pkl')
        with open(stacker_model_path, 'wb') as f:
            pickle.dump(model_dict['stacker'], f)

        feature_info_path = os.path.join(feature_dir, 'feature_info.pkl')
        with open(feature_info_path, 'wb') as f:
            pickle.dump({
                'feature_slice': model_dict['feature_slice'],
                'performance': model_dict['performance']
            }, f)

        print(f"  - Pearson r: {model_dict['performance']['pearson_r']:.3f}")

summary_path = os.path.join('saved_models', 'model_summary.pkl')
with open(summary_path, 'wb') as f:
    pickle.dump(performance_summary, f)