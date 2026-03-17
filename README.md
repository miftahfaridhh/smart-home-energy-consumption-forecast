# Smart Home Energy Consumption Forecasting

Comparative study of classical time-series and deep learning models for forecasting household energy consumption using smart meter data with weather features. Research project at Kookmin University, Seoul.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project investigates the effectiveness of various forecasting models for predicting smart home energy consumption. Using real-world smart meter data combined with weather information, we compare classical statistical methods (ARIMA, SARIMA, SARIMAX) against deep learning approaches (LSTM, BiLSTM, Stacked BiLSTM, RNN, GRU) to determine the optimal approach for energy demand forecasting.

**Key Finding**: LSTM Multivariate significantly outperforms all classical methods, achieving the lowest MSE (0.022) and MAE (0.110), demonstrating that deep learning models with weather features as multivariate input provide substantially better forecasting accuracy.

## Dataset

- **Source**: [Smart Home Dataset with Weather Information](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information) (Kaggle)
- **Records**: 503,910 entries at minute-level granularity
- **Period**: January 1, 2016 — December 16, 2016
- **Energy Features**: House overall, Dishwasher, Furnace, Home office, Fridge, Wine cellar, Garage door, Kitchen, Barn, Well, Microwave, Living room, Solar
- **Weather Features**: Temperature, humidity, apparent temperature, pressure, cloud cover, wind bearing, precipitation intensity, dew point, precipitation probability

## Methodology

### Data Preprocessing

1. **Missing Value Handling** — Quadratic interpolation for null values
2. **Normalization** — MinMax scaling to [-1, 1] range
3. **Feature Selection** — Correlation analysis to select the most relevant weather features
4. **Outlier Removal** — Statistical outlier detection and deletion
5. **Windowing** — Sliding window transformation for sequence-to-sequence learning
6. **Data Split** — 80% training / 20% validation with separate test set

### Model Architectures

| Model | Type | Description |
|---|---|---|
| ARIMA Basic | Statistical | Autoregressive Integrated Moving Average |
| ARIMA Dynamic | Statistical | ARIMA with dynamic predictions |
| SARIMA | Statistical | Seasonal ARIMA |
| SARIMAX | Statistical | SARIMA with exogenous variables (weather) |
| LSTM Univariate | Deep Learning | Single-feature LSTM |
| LSTM Multivariate | Deep Learning | Multi-feature LSTM with weather inputs |
| Stacked BiLSTM | Deep Learning | 2-layer Bidirectional LSTM |
| BiLSTM | Deep Learning | Single Bidirectional LSTM layer |
| RNN | Deep Learning | Simple Recurrent Neural Network |
| GRU | Deep Learning | Gated Recurrent Unit |

### Training Configuration

- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 0.001)
- **Epochs**: 200 (with early stopping)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Activation**: tanh

## Results

### Smart Home Dataset (Kaggle)

| Model | MSE | RMSE | MAE | R2 |
|---|---|---|---|---|
| ARIMA Basic | 0.259 | 0.509 | 0.463 | 0.722 |
| ARIMA Dynamic | 0.069 | 0.263 | 0.176 | 0.229 |
| SARIMA | 0.107 | 0.327 | 0.266 | 0.397 |
| SARIMAX | 0.101 | 0.317 | 0.243 | 0.363 |
| LSTM Univariate | 0.068 | 0.261 | 0.173 | 0.307 |
| **LSTM Multivariate** | **0.022** | **0.150** | **0.110** | **0.173** |

### South Korean Regional Power Data (17 Regions)

Extended experiments on regional power consumption data across 17 South Korean regions (Seoul, Busan, Gangwon, Daejeon, Incheon, Gyeonggi, Gwangju, Jeju, Daegu, etc.) using 5 deep learning models:

| Model | MSE | RMSE | MAE | R2 |
|---|---|---|---|---|
| Stacked BiLSTM | — | — | — | — |
| BiLSTM | 0.0143 | 0.1196 | 0.0612 | 0.9549 |
| LSTM | — | — | — | — |
| RNN | — | — | — | — |
| GRU | — | — | — | — |

*Example: BiLSTM results on Busan regional data. R2 = 0.9549 indicates excellent prediction accuracy.*

## Project Structure

```
smart-home-energy-consumption-forecast/
├── 1.1 Preprocessing From Dataset.ipynb     # Data loading, cleaning, interpolation
├── 1.2 Preprocessing MinMax and Windowing.ipynb  # Normalization, sliding window
├── 1.3.1 Stacked_BiLSTM_Train.ipynb         # Stacked BiLSTM training
├── 1.3.2 BiLSTM_Train.ipynb                 # BiLSTM training
├── 1.3.3 LSTM_Train.ipynb                   # LSTM training
├── 1.3.4 RNN_Train.ipynb                    # RNN training
├── 1.3.5 GRU_Train.ipynb                    # GRU training
├── 1.4 Mixed Model.ipynb                    # Model comparison on regional data
├── prepare.ipynb                            # Additional data preparation
├── smart-home-iot-eda-arimas-lstm-and-more.ipynb  # EDA + classical models
├── dataset/                                 # Raw data files
├── checkpoint/                              # Saved model weights
├── tools/
│   └── preprocess_lib.py                    # Preprocessing utilities
└── Paper Smart Home/
    └── Fixed.docx                           # Research paper
```

## Requirements

- Python >= 3.8
- TensorFlow >= 2.x
- pandas
- NumPy
- scikit-learn
- statsmodels
- matplotlib
- seaborn

## Installation & Usage

```bash
git clone https://github.com/miftahfaridhh/smart-home-energy-consumption-forecast.git
cd smart-home-energy-consumption-forecast
pip install tensorflow pandas numpy scikit-learn statsmodels matplotlib seaborn
```

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information) and place it in `dataset/`
2. Run notebooks sequentially starting from `1.1 Preprocessing From Dataset.ipynb`

## Evaluation Metrics

| Metric | Formula | Description |
|---|---|---|
| **MSE** | Mean Squared Error | Average squared prediction error |
| **RMSE** | Root Mean Squared Error | Square root of MSE |
| **MAE** | Mean Absolute Error | Average absolute prediction error |
| **R2** | Coefficient of Determination | Proportion of variance explained |
| **MAPE** | Mean Absolute Percentage Error | Percentage-based error metric |
| **SMAPE** | Symmetric MAPE | Symmetric percentage error |

## Citation

> Muhammad Miftah Faridh, Yeong Min Jang. "Smart Home Power Consumption Forecasting." Department of Electronics Engineering, Kookmin University, Seoul, South Korea.

## License

This project is open-sourced under the [MIT License](LICENSE).
