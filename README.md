

# VO-evolution

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![MeteoCat](https://img.shields.io/badge/API-MeteoCat-005A9E?style=for-the-badge)

![Banner](https://raw.githubusercontent.com/DavidDDRC99/VO-evolution/main/banner.png)

Anàlisi climàtica del **Vallès Occidental** (Sabadell, Vacarisses) als últims ~30 anys. Descarrega, processa i analitza dades meteorològiques de l'API de [MeteoCat](https://www.meteo.cat/) per detectar tendències de temperatura, precipitació i altres variables.

---

## Descripció

VO-evolution analitza dades de **3 estacions meteorològiques**:

| Estació | Periodicitat | Període |
|---------|-------------|---------|
| Sabadell Centre | Diària | 2008–2026 |
| Sabadell Nord (Parc Agrari) | 30 min | 2008-10-24 → 2026-04-01 |
| Vacarisses | 30 min | 1996-02-16 → 2026-04-01 |

**Variables**: temperatura mitjana/màxima/mínima, humitat, precipitació, vent, pressió, radiació.

---

## Tecnologies

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) **Python 3.10+** · ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) **Pandas/NumPy** · ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white) **SciPy** · ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white) **Matplotlib** · ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) **Jupyter** · ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) **Flask**

---

## Estructura

`
VO-evolution/
├── scripts/                              # Descàrrega i processament
├── Seccions/                             # Anàlisis (Pluges, Temperatures, vent, Correlacions)
├── dashboard/                            # Dashboard web (Flask)
├── Cleaned Data/                         # CSV nets
├── Raw Data/                             # Dades originals
├── Data collection.ipynb                 # Notebook de descàrrega
└── data_preparation.ipynb                # Neteja i concatenació
`

---

## Instal·lació

`ash
git clone https://github.com/DavidDDRC99/VO-evolution.git
cd VO-evolution
pip install pandas numpy scipy matplotlib seaborn jupyter requests flask
`

## Ús

`ash
# Descarregar dades
python scripts/download_two_stations_simple.py

# Dashboard
cd dashboard && python app.py   # http://127.0.0.1:5000

# Tests estadístics
python scripts/inject_stat_tests.py
`

## Anàlisis

- **Pluges** — Precipitació mensual, horària, intensitat i patrons estacionals
- **Temperatures** — Evolució de temperatures nocturnes a l'estiu
- **Vent** — Direcció i velocitat
- **Correlacions** — Matriu de correlacions entre variables
- **Estadística** — Mann-Kendall, Sen's slope, Pettitt change-point detection

## Dashboard

Dashboard interactiu (Flask) per visualitzar gràfics i explorar dades al navegador.

## Resultats

- Escalfament en temperatures nocturnes d'estiu
- Canvis en patró de pluges amb variació estacional
- Correlació temperatura-humitat

## Autor

**David Domínguez Ruiz** · [LinkedIn](https://www.linkedin.com/in/david-dominguez-ruiz-8720961a3) · [GitHub](https://github.com/DavidDDRC99) · davidddrc99@gmail.com

---

*Dades obertes del [Servei Meteorològic de Catalunya (MeteoCat)](https://www.meteo.cat/)*
