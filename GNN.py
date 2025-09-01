import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from spektral.layers import GraphConv
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

def build_knn_graph(X, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, indices = nn.kneighbors(X)
    rows = []
    cols = []
    for i in range(indices.shape[0]):
        for j in range(1, k + 1):
            rows.append(i)
            cols.append(indices[i, j])
    all_rows = rows + cols
    all_cols = cols + rows
    all_values = [1.0] * len(all_rows)
    adj_matrix = coo_matrix((all_values, (all_rows, all_cols)), shape=(X.shape[0], X.shape[0]))
    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix[adj_matrix > 0] = 1.0
    return adj_matrix

def build_gnn(input_node_features_shape, input_adj_shape):
    X_in = Input(shape=(input_node_features_shape,))
    A_in = Input(shape=(input_adj_shape,), sparse=True)
    x = GraphConv(64, activation='relu')([X_in, A_in])
    x = Dropout(0.1)(x)
    x = GraphConv(32, activation='relu')([x, A_in])
    x = Dropout(0.1)(x)
    output = Dense(1, activation='linear')(x)
    model = Model(inputs=[X_in, A_in], outputs=output)
    return model

fig, axes = plt.subplots(9, 1, figsize=(7, 36))

if axes.ndim == 1:
    axes = axes.reshape(-1, 1)

global_adj_matrix = build_knn_graph(data, k=5)

for t_idx, target_name in enumerate(target_names):
    y = x[target_name].to_numpy()
    for f_idx, (feat_slice, feat_title) in enumerate(feature_sets):
        row_idx = t_idx * 3 + f_idx
        X = data[:, feat_slice]
        X_scaled = X / np.maximum(X.max(axis=0), 1e-8)

        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X_scaled, y, np.arange(X_scaled.shape[0]), test_size=0.1, random_state=42
        )

        model_gnn = build_gnn(X_train.shape[1], global_adj_matrix.shape)
        model_gnn.compile(optimizer=Adam(learning_rate=0.005), loss='mean_squared_error')

        adj_sparse_tf = tf.SparseTensor(
            indices=np.array([global_adj_matrix.row, global_adj_matrix.col]).T,
            values=global_adj_matrix.data,
            dense_shape=global_adj_matrix.shape
        )
        model_gnn.fit(
            [X_train, adj_sparse_tf], y_train,
            batch_size=32,
            epochs=100,
            verbose=0
        )

        y_pred_gnn = model_gnn.predict([X_test, adj_sparse_tf]).flatten()

        p_r_gnn, _ = pearsonr(y_pred_gnn, y_test)
        mse_gnn = mean_squared_error(y_test, y_pred_gnn)
        rmse_gnn = np.sqrt(mse_gnn)

        ax_gnn = axes[row_idx, 0]
        sns.regplot(x=y_pred_gnn, y=y_test, ax=ax_gnn, scatter_kws={'s': 25, 'alpha': 0.7})
        ax_gnn.set_xlabel('Predicted Value')
        ax_gnn.set_ylabel('True Value')
        ax_gnn.set_title(f"{feat_title} | {target_name}\nGNN (kNN Graph)")
        ax_gnn.text(
            0.05, 0.95,
            f"Pearson Correlation={p_r_gnn:.3f}\nRMSE={rmse_gnn:.2f}",
            ha='left', va='top',
            transform=ax_gnn.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

plt.tight_layout()
plt.show()
