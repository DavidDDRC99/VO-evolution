![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![MeteoCat](https://img.shields.io/badge/API-MeteoCat-005A9E?style=for-the-badge)

![Banner](https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/main/banner.png)

Anàlisi climàtica del **Vallès Occidental** (Sabadell, Vacarisses, Terrassa) durant els últims ~30 anys. El projecte descarrega dades meteorològiques de les estacions de [MeteoCat](https://www.meteo.cat/), les neteja, i les analitza des de múltiples perspectives: pluges, temperatures, sequera, correlacions entre variables, i downscaling mitjançant machine learning.

---

## Estacions i Dades

| Estació | Codi | Resolució | Període | Variables |
|---------|------|-----------|---------|-----------|
| **Sabadell Centre** | — | Diària | 2008–2026 | T, precipitació, vent |
| **Sabadell Nord** (Parc Agrari) | `XF` | 30 min | 2008–2026 | T, HR, precipitació, vent, pressió, radiació |
| **Vacarisses** | `D2` | 30 min | 1996–2026 | T, HR, precipitació, vent, pressió, radiació |

**Font:** [Meteocat](https://www.meteo.cat) — Servei Meteorològic de Catalunya.

---

## Anàlisis

### 🌧 Pluges (`Seccions/Pluges/`)
- **Rain_seasons.ipynb** — Evolució mensual de la precipitació per les 3 estacions. Tests de tendència (Mann-Kendall, Sen's slope, Pettitt).
- **Hourly_rain_analysis.ipynb** — Quines hores del dia plou més? Patrons horaris a Vacarisses i Sbd Nord.
- **Intensity_pluja.ipynb** — Intensitat de la pluja a resolució 30 min: histogrames, màxims anuals, pluja específica mitjana anual.

### 🌡 Temperatures (`Seccions/Temperatures/`)
- **Evolucio_temperatures_nocturnes.ipynb** — Evolució de les temperatures nocturnes i diürnes. Classificació de nits (tropicals/tòrrides/infernals). Comparació Centre vs Nord.
- **Nits_i_duracio_estacions.ipynb** — Expansió de l'estiu i contracció de l'hivern: quants dies s'ha allargat l'estiu als últims 18 anys?

### 🏜 Sequera (`Seccions/Sequera/`)
- **Drought_analysis.ipynb** — Anàlisi de sequera amb índex **SPEI** (Standardized Precipitation Evapotranspiration Index). PET calculat amb el mètode de Thornthwaite (1948). Cobreix les 3 estacions i múltiples escales temporals.

### 🔗 Correlacions
- **Correlacions_de_variables.ipynb** — Correlacions Pearson i Spearman entre totes les variables meteorològiques (T, HR, precipitació, pressió, vent) a Sbd Nord i Vacarisses.

### 🤖 Machine Learning: Downscaling de Temperatura (`Seccions/ML/`)
- **Prediccio_corba_temperatura.ipynb** — Predicció de la corba de temperatura a resolució 30 min (48 valors/dia) a partir de dades diàries.
- **Models provats:** Sinusoïdal (baseline), KNN, Random Forest, XGBoost, **MLP** (millor).
- **Millor model:** MLP amb `(256, 128)` neurones, `lr=0.01`, `alpha=0.001` → **RMSE 1.177°C** (R² 0.972).
- **Transfer learning a Sbd Centre:** S'ha generat `Cleaned Data/Sbd_Centre_hourly_synthetic.csv` — sèrie horària sintètica de temperatura per Sbd Centre (279.984 files, 2008–2026) utilitzant el model entrenat a Sbd Nord amb les 12 variables comunes. RMSE 1.149°C al test set de validació.

---

## Dashboard Interactiu

El directori `dashboard/` conté una aplicació **Dash/Plotly** amb 3 pestanyes:

| Pestanya | Funcionalitats |
|----------|---------------|
| **🌧 Pluja** | Mitjana mensual, evolució anual (graella 4×3), intensitat horària, pluja específica. Selector: Centre/Nord/Vacarisses |
| **🌡 Temperatura** | Nits càlides per llindars, ratxes de calor, comparació T_min, durada estiu/hivern. Selector: Nord/Centre |
| **💨 Vent** | Top 20 dies més ventats, histograma horari, heatmaps hora/any i mes/any, distribució mensual |

**Per llançar-lo:**
```bash
cd dashboard
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
python app.py
```

O fent doble clic a `run_dashboard.bat` (Windows) o `run_dashboard.command` (Mac).

---

## Estructura del Projecte

```
VO-evolution/
├── Data collection.ipynb         # Descarrega dades crues de Meteocat
├── data_preparation.ipynb        # Neteja i concatenació de dades
├── Seccions/                     # Notebooks d'anàlisi (7 en total)
│   ├── Pluges/                   # Anàlisi de pluges (3 notebooks)
│   ├── Temperatures/             # Anàlisi de temperatures (2 notebooks)
│   ├── ML/                       # Downscaling ML (2 notebooks)
│   ├── Sequera/                  # Anàlisi de sequera (1 notebook)
│   └── Correlacions_de_variables.ipynb
├── scripts/                      # Mòduls Python reutilitzables
│   ├── statistical_tests.py      # Mann-Kendall, Sen's slope, Pettitt
│   ├── drought_analysis.py       # SPI/SPEI drought indices
│   ├── inject_stat_tests.py      # Injecta tests als notebooks
│   ├── prepare_ml_data.py        # Feature engineering per ML
│   ├── generate_ml_notebook.py   # Genera notebook ML automàticament
│   ├── generate_sbd_centre_hourly.py  # Genera dada horària sintètica
│   └── validate_synthetic.py     # Quality checks
├── dashboard/                    # Aplicació Dash interactiva
│   ├── app.py                    # Punt d'entrada
│   ├── data_loader.py            # Càrrega de dades
│   ├── plots_rain.py             # Visualitzacions de pluja
│   ├── plots_temperature.py      # Visualitzacions de temperatura
│   ├── plots_wind.py             # Visualitzacions de vent
│   ├── utils.py                  # Funcions compartides
│   ├── run_dashboard.bat         # Llançador Windows
│   └── run_dashboard.command     # Llançador Mac
├── tests/                        # Tests unitaris (pytest)
│   ├── test_utils.py             # 7 tests per dashboard/utils.py
│   └── test_statistical.py       # 9 tests per statistical_tests.py
├── requirements.txt
├── AGENTS.md                     # Context per a l'agent opencode
└── README.md
```

---

## Com Reproduir

1. **Clonar el repositori:**
   ```bash
   git clone https://github.com/DavidDDRC99/VO-evolution.git
   cd VO-evolution
   ```

2. **Instal·lar dependències:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Descarregar dades crues:**
   ```bash
   # Opció A: Descarrega completa (~2.5h)
   python scripts/download_two_stations_simple.py
   # Opció B: Pas a pas amb notebooks
   # Obrir Data collection.ipynb al Jupyter
   ```

4. **Processar i analitzar:**
   ```bash
   jupyter notebook Seccions/Pluges/
   jupyter notebook Seccions/Temperatures/
   jupyter notebook Seccions/Sequera/
   jupyter notebook Seccions/ML/
   ```

5. **Tests:**
   ```bash
   python -m pytest tests/
   ```

---

## Resultats Clau

| Àmbit | Troballa principal |
|-------|-------------------|
| **Tendència pluges** | Tests Mann-Kendall sobre la precipitació mensual a les 3 estacions |
| **Nits càlides** | Augment de nits tropicals (>20°C) al Centre respecte al Nord |
| **Expansió estiu** | L'estiu s'ha allargat ~X dies als últims 18 anys |
| **Downscaling ML** | MLP prediu la corba de temperatura 30 min amb RMSE 1.15–1.18°C |
| **Sequera** | SPEI a múltiples escales temporals per les 3 estacions |

---

## Autor

**David Domínguez Ruiz** · [LinkedIn](https://www.linkedin.com/in/david-dominguez-ruiz-8720961a3) · [GitHub](https://github.com/DavidDDRC99) · davidddrc99@gmail.com

*Dades obertes del [Servei Meteorològic de Catalunya (MeteoCat)](https://www.meteo.cat/)*
