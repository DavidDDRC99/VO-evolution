import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3.12.0"}
}

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

md("# Predicció de la Corba de Temperatura (30 min) a partir de Dades Diàries\n"
   "\n"
   "**Objectiu:** Donats els agregats diaris de temperatura (T_min, T_max, T_avg) d'un dia, "
   "predir la corba de temperatura d'aquell mateix dia a resolució 30 min (48 valors).\n"
   "\n"
   "**Enfocament:** Regressió multi-sortida amb 5 models:\n"
   "1. **Sinusoïdal** (baseline físic)\n"
   "2. **KNN** (veïns)\n"
   "3. **Random Forest**\n"
   "4. **XGBoost**\n"
   "5. **MLP** (xarxa neuronal)\n"
   "\n"
   "Tots els models ML utilitzen `Pipeline` + `GridSearchCV` amb `TimeSeriesSplit`.")

md("## 1. Imports i Configuració")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import curve_fit

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost no instal·lat — s'ometrà el model XGBoost")

HOURS_PER_DAY = 48
DATA_PATH = Path("Cleaned Data") / "ml_data.npz"
META_PATH = Path("Cleaned Data") / "ml_metadata.csv"
RANDOM_STATE = 42""")

md("## 2. Càrrega de Dades")

code("""data = np.load(DATA_PATH, allow_pickle=False)
X = data['X']
y = data['y']
feature_names = data['feature_names'].tolist()
target_names = data['target_names'].tolist()
years = data['years']
months = data['months']
days = data['days']
station_ids = data['station_ids']

meta = pd.read_csv(META_PATH)
dates = meta['date'].values
stations = meta['station'].values
timestamps = pd.to_datetime(meta['date_obj'])

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Features: {feature_names}")
print(f"Targets: {target_names[0]} .. {target_names[-1]}")
print(f"Date range: {timestamps.min().date()} to {timestamps.max().date()}")

df_meta = pd.DataFrame({'date': dates, 'station': stations, 'timestamp': timestamps})
df_meta['year'] = timestamps.dt.year
df_meta['month'] = timestamps.dt.month
print(f"\\nSamples per station:")
print(df_meta['station'].value_counts())
print(f"\\nSamples per year:")
print(df_meta['year'].value_counts().sort_index())""")

md("## 3. Train/Test Split Temporal\n"
   "Utilitzem dades fins 2020 per entrenar i de 2021 en endavant per testar.")

code("""train_mask = timestamps.dt.year <= 2020
test_mask = ~train_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
dates_test = dates[test_mask]
stations_test = stations[test_mask]
timestamps_train = timestamps[train_mask]
timestamps_test = timestamps[test_mask]
years_test = years[test_mask]
months_test = months[test_mask]

train_years = sorted(timestamps_train.dt.year.unique())
test_years = sorted(timestamps_test.dt.year.unique())

print(f"Train: {X_train.shape[0]:,} dies ({train_years[0]}-{train_years[-1]})")
print(f"Test:  {X_test.shape[0]:,} dies ({test_years[0]}-{test_years[-1]})")""")

md("## 4. Visualització Exploratòria")

code("""slots = np.arange(HOURS_PER_DAY)
hours = slots / 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, station_name in zip(axes, ['Sbd Nord', 'Vacarisses']):
    mask = stations_test == station_name
    mean_curve = y_test[mask].mean(axis=0)
    std_curve = y_test[mask].std(axis=0)
    ax.plot(hours, mean_curve, 'b-', linewidth=2)
    ax.fill_between(hours, mean_curve - std_curve, mean_curve + std_curve,
                     alpha=0.2, color='b')
    ax.set_title(f"{station_name} (test set, {mask.sum()} dies)")
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 25, 3))

fig.suptitle("Corba de Temperatura Mitjana (Test Set)", fontsize=14)
plt.tight_layout()
plt.show()""")

code("""season_map = {'Hivern': [12, 1, 2], 'Primavera': [3, 4, 5],
              'Estiu': [6, 7, 8], 'Tardor': [9, 10, 11]}
colors_season = {'Hivern': 'blue', 'Primavera': 'green', 'Estiu': 'red', 'Tardor': 'orange'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, station_name in zip(axes, ['Sbd Nord', 'Vacarisses']):
    mask = stations_test == station_name
    for season_name, mlist in season_map.items():
        smask = mask & (np.isin(pd.to_datetime(dates_test).month, mlist))
        if smask.sum() == 0:
            continue
        curve = y_test[smask].mean(axis=0)
        ax.plot(hours, curve, label=f"{season_name} (n={smask.sum()})",
                color=colors_season[season_name], linewidth=2)
    ax.set_title(station_name)
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("Temperatura (°C)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 25, 3))

fig.suptitle("Corba Mitjana per Estació de l'Any (Test Set)", fontsize=14)
plt.tight_layout()
plt.show()""")

md("## 5. Funcions d'Avaluació")

code("""def evaluate_model(name, y_true, y_pred, verbose=True):
    rmse_per_slot = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    if verbose:
        print(f"{name}:")
        print(f"  RMSE global: {overall_rmse:.3f} °C")
        print(f"  RMSE per slot: {rmse_per_slot.mean():.3f} ± {rmse_per_slot.std():.3f} °C")
        print(f"  R²: {r2:.4f}")
        print(f"  Bias: {bias:.3f} °C")
    return {'name': name, 'rmse_global': overall_rmse, 'rmse_mean': rmse_per_slot.mean(),
            'rmse_std': rmse_per_slot.std(), 'r2': r2, 'bias': bias,
            'rmse_per_slot': rmse_per_slot}

def plot_curves(y_true, y_pred, title, n_examples=4, dates=None, stations=None):
    np.random.seed(RANDOM_STATE)
    idxs = np.random.choice(len(y_true), n_examples, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    hours_display = np.arange(HOURS_PER_DAY) / 2
    for ax, idx in zip(axes.flat, idxs):
        ax.plot(hours_display, y_true[idx], 'b-o', label='Real', markersize=3, linewidth=1.5)
        ax.plot(hours_display, y_pred[idx], 'r--s', label='Predicció', markersize=3, linewidth=1.5)
        label = ""
        if dates is not None:
            label += f"{dates[idx]} "
        if stations is not None:
            label += f"({stations[idx]})"
        ax.set_title(label)
        ax.set_xlabel("Hora")
        ax.set_ylabel("Temperatura (°C)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(0, 25, 3))
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_rmse_per_slot(results):
    fig, ax = plt.subplots(figsize=(12, 5))
    hours_display = np.arange(HOURS_PER_DAY) / 2
    for res in results:
        ax.plot(hours_display, res['rmse_per_slot'],
                label=f"{res['name']} (global: {res['rmse_global']:.3f}°C)",
                linewidth=1.5)
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("RMSE per Franja Horària (30 min)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 25, 3))
    plt.tight_layout()
    plt.show()

def results_table(results):
    print(f"{'Model':<20} {'RMSE (°C)':<12} {'R²':<10} {'Bias (°C)':<10}")
    print("-" * 52)
    for res in results:
        print(f"{res['name']:<20} {res['rmse_global']:<12.3f} {res['r2']:<10.4f} {res['bias']:<10.3f}")""")

md("## 6. Model 1: Baseline Sinusoïdal\n"
   "Ajustem una corba sinusoïdal a cada dia: $T(t) = A \\sin(\\omega t + \\phi) + T_{avg}$ "
   "on $A \\approx (T_{max} - T_{min})/2$ i optimitzem la fase per dia.")

code("""def fit_baseline(X_test, y_test):
    y_pred = np.zeros_like(y_test)
    t = np.arange(HOURS_PER_DAY) / 2
    for i in range(len(y_test)):
        T_min, T_max, T_avg = X_test[i, 0], X_test[i, 1], X_test[i, 2]
        A_init = max((T_max - T_min) / 2, 0.1)
        try:
            popt, _ = curve_fit(
                lambda x, A, phi: T_avg + A * np.sin(2*np.pi*x/24 + phi),
                t, y_test[i], p0=[A_init, -np.pi/2 + 3.5], maxfev=5000
            )
            y_pred[i] = T_avg + popt[0] * np.sin(2*np.pi*t/24 + popt[1])
        except:
            y_pred[i] = T_avg + A_init * np.sin(2*np.pi*t/24 - np.pi/2 + 3.5)
    return y_pred

print("Ajustant baseline sinusoïdal...")
y_pred_bl = fit_baseline(X_test, y_test)
res_bl = evaluate_model("Sinusoïdal", y_test, y_pred_bl)
plot_curves(y_test, y_pred_bl, "Baseline Sinusoïdal: Predicció vs Real",
            dates=dates_test, stations=stations_test)""")

md("## 7. Model 2: KNN")

code("""knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])

knn_params = {
    'knn__n_neighbors': [5, 10, 20, 30, 50],
    'knn__weights': ['distance', 'uniform'],
    'knn__p': [1, 2]
}

tscv = TimeSeriesSplit(n_splits=5)
print("Cerca GridSearch per KNN...")
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=tscv,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
knn_grid.fit(X_train, y_train)

print(f"\\nMillors paràmetres KNN: {knn_grid.best_params_}")
y_pred_knn = knn_grid.predict(X_test)
res_knn = evaluate_model("KNN", y_test, y_pred_knn)

plot_curves(y_test, y_pred_knn,
            f"KNN (k={knn_grid.best_params_['knn__n_neighbors']}): Predicció vs Real",
            dates=dates_test, stations=stations_test)""")

md("## 8. Model 3: Random Forest")

code("""rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
])

rf_params = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_leaf': [1, 2, 5]
}

print("\\nCerca GridSearch per Random Forest...")
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=tscv,
                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

print(f"\\nMillors paràmetres RF: {rf_grid.best_params_}")
y_pred_rf = rf_grid.predict(X_test)
res_rf = evaluate_model("Random Forest", y_test, y_pred_rf)

plot_curves(y_test, y_pred_rf, "Random Forest: Predicció vs Real",
            dates=dates_test, stations=stations_test)""")

md("## 9. Model 4: XGBoost")

code("""if HAS_XGB:
    xgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', MultiOutputRegressor(
            XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
        ))
    ])

    xgb_params = {
        'xgb__estimator__n_estimators': [100, 200],
        'xgb__estimator__max_depth': [4, 6, 8],
        'xgb__estimator__learning_rate': [0.05, 0.1],
        'xgb__estimator__subsample': [0.8, 1.0]
    }

    print("\\nCerca GridSearch per XGBoost...")
    xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=tscv,
                            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, y_train)

    print(f"\\nMillors paràmetres XGBoost: {xgb_grid.best_params_}")
    y_pred_xgb = xgb_grid.predict(X_test)
    res_xgb = evaluate_model("XGBoost", y_test, y_pred_xgb)
    plot_curves(y_test, y_pred_xgb, "XGBoost: Predicció vs Real",
                dates=dates_test, stations=stations_test)
else:
    print("XGBoost no disponible — s'omet")
    y_pred_xgb = None""")

md("## 10. Model 5: MLP (Xarxa Neuronal)")

code("""mlp_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(random_state=RANDOM_STATE, early_stopping=True, max_iter=500))
])

mlp_params = {
    'mlp__hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128)],
    'mlp__learning_rate_init': [0.001, 0.01],
    'mlp__alpha': [0.0001, 0.001]
}

print("\\nCerca GridSearch per MLP...")
mlp_grid = GridSearchCV(mlp_pipe, mlp_params, cv=tscv,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
mlp_grid.fit(X_train, y_train)

print(f"\\nMillors paràmetres MLP: {mlp_grid.best_params_}")
y_pred_mlp = mlp_grid.predict(X_test)
res_mlp = evaluate_model("MLP", y_test, y_pred_mlp)

plot_curves(y_test, y_pred_mlp, "MLP: Predicció vs Real",
            dates=dates_test, stations=stations_test)""")

md("## 11. Comparació de Models")

code("""results = [res_bl, res_knn, res_rf]
if HAS_XGB:
    results.append(res_xgb)
results.append(res_mlp)

results_table(results)""")

code("""plot_rmse_per_slot(results)""")

code("""names = [r['name'] for r in results]
rmse_vals = [r['rmse_global'] for r in results]
r2_vals = [r['r2'] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

bars1 = axes[0].bar(names, rmse_vals, color=colors_bar[:len(names)])
axes[0].set_ylabel("RMSE Global (°C)")
axes[0].set_title("RMSE per Model (Test Set)")
axes[0].grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=9)

bars2 = axes[1].bar(names, r2_vals, color=colors_bar[:len(names)])
axes[1].set_ylabel("R²")
axes[1].set_title("R² per Model (Test Set)")
axes[1].grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, r2_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()""")

md("### Error per Estació i Estació de l'Any")

code("""# Error per estació
pred_dict = {}
for name, pred in zip(['Sinusoïdal', 'KNN', 'Random Forest'],
                       [y_pred_bl, y_pred_knn, y_pred_rf]):
    pred_dict[name] = pred
if HAS_XGB:
    pred_dict['XGBoost'] = y_pred_xgb
pred_dict['MLP'] = y_pred_mlp

station_errors = {}
for station_name in ['Sbd Nord', 'Vacarisses']:
    mask = stations_test == station_name
    station_errors[station_name] = []
    for name, pred in pred_dict.items():
        rmse = np.sqrt(mean_squared_error(y_test[mask], pred[mask]))
        station_errors[station_name].append(rmse)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(pred_dict))
width = 0.35
ax.bar(x - width/2, station_errors['Sbd Nord'], width, label='Sbd Nord', color='steelblue')
ax.bar(x + width/2, station_errors['Vacarisses'], width, label='Vacarisses', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(pred_dict.keys())
ax.set_ylabel("RMSE (°C)")
ax.set_title("Error per Estació i Model")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()""")

code("""season_errors = {}
for season_name, mlist in season_map.items():
    mask = np.isin(months_test, mlist)
    season_errors[season_name] = []
    for name, pred in pred_dict.items():
        rmse = np.sqrt(mean_squared_error(y_test[mask], pred[mask]))
        season_errors[season_name].append(rmse)

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(pred_dict))
width = 0.2
season_colors = ['blue', 'green', 'red', 'orange']
for i, (season_name, color) in enumerate(zip(season_map.keys(), season_colors)):
    ax.bar(x + (i - 1.5) * width, season_errors[season_name], width,
           label=season_name, color=color, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(pred_dict.keys())
ax.set_ylabel("RMSE (°C)")
ax.set_title("Error per Estació de l'Any i Model")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()""")

md("### Importància de les Features (Random Forest)")

code("""rf_best = rf_grid.best_estimator_.named_steps['rf']
importances = rf_best.feature_importances_
idx_sort = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(importances)), importances[idx_sort])
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([feature_names[i] for i in idx_sort])
ax.set_xlabel("Importància")
ax.set_title("Importància de les Features (Random Forest)")
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()""")

md("## 12. Conclusions\n"
   "Resum de resultats (omplir després d'executar):\n"
   "- **Millor model:**\n"
   "- **RMSE assolit:**\n"
   "- **Limitacions:**\n"
   "- **Properes passes:**")

nb.cells = cells

out_path = "Seccions/ML/Prediccio_corba_temperatura.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook created: {out_path}")
