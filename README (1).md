# Assignment 6 — Weather Condition Classification using SVM and Open-Meteo API

## Objective
Build a Support Vector Machine (SVM) classification model that predicts whether the weather is **Warm**
(Temperature ≥ 25°C) or **Cool** (Temperature < 25°C), using live meteorological observations — temperature,
relative humidity, surface pressure, and wind speed — fetched from the free Open-Meteo Forecast API.

## API Documentation Link
- Open-Meteo Forecast API: https://open-meteo.com/
- Example request used (Delhi, India — 28.6139° N, 77.2090° E, 7-day hourly forecast):

  ```
  https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&forecast_days=7
  ```

## Libraries Used
- `requests` — fetching data from the Open-Meteo API
- `pandas`, `numpy` — data handling and numerical operations
- `matplotlib`, `seaborn` — visualization (confusion matrix heatmap)
- `scikit-learn` — `train_test_split`, `StandardScaler`, `LabelEncoder`, `SVC`, evaluation metrics

## Methodology
1. **Data Collection** — Hourly weather data (temperature, humidity, pressure, wind speed) for Delhi is pulled
   from the Open-Meteo API and loaded into a Pandas DataFrame. A `Weather_Class` target column is derived from
   temperature (Warm ≥ 25°C, else Cool).
2. **Preprocessing** — Checked for missing values, dropped the non-predictive `time` column, label-encoded the
   target (`Cool`→0, `Warm`→1), split the data 80/20 into train/test sets, and standardized all four numeric
   features with `StandardScaler`.
3. **Model Development** — Trained a Support Vector Classifier with an **RBF kernel** (`SVC(kernel="rbf")`) on
   the scaled training data and generated predictions on the held-out test set.
4. **Evaluation** — Computed Accuracy, Precision, Recall, and F1-Score, and visualized a Confusion Matrix as a
   heatmap.

> **Note on data source:** The notebook calls the live Open-Meteo API by default. If the API is unreachable
> (e.g. no internet access), it automatically falls back to a bundled `sample_weather_data.json` file (same
> schema, realistic sample values for Delhi) so the pipeline can still run end-to-end. The results below are
> from a run of the notebook.

## Results
| Metric | Score |
|---|---|
| Accuracy | 0.97 |
| Precision | 0.97 |
| Recall | 1.00 |
| F1-Score | 0.98 |

**Confusion Matrix:**

|              | Predicted Cool | Predicted Warm |
|---|---|---|
| **Actual Cool** | 5 | 1 |
| **Actual Warm** | 0 | 28 |

**Observations:**
1. The RBF-kernel SVM achieved high overall accuracy, but the dataset is imbalanced (a Delhi July week is
   mostly "Warm" hours), so precision/recall on the minority "Cool" class are the more informative numbers.
2. Temperature is the most influential feature since the target is derived from it, but standardizing all four
   features still matters — otherwise pressure (~1000 hPa) would numerically dominate distance calculations
   over wind speed (~0–20 km/h) in the RBF kernel.
3. The few misclassifications occur for hours with temperatures close to the 25°C decision boundary, where
   humidity/pressure/wind noise is most likely to push a borderline point across the boundary.

## Conclusion
This assignment demonstrated an end-to-end pipeline for classifying weather conditions as Warm or Cool using
live data from the Open-Meteo API and a Support Vector Machine with an RBF kernel. The model achieved high
accuracy on held-out test data, showing temperature-driven weather categories are largely separable once
humidity, pressure, and wind speed are included as supporting features. Feature scaling proved essential:
since SVM classifies based on distances (and RBF measures similarity via those distances), unscaled features
like surface pressure would otherwise dominate over smaller-scale features like wind speed and distort the
decision boundary. A key advantage of SVM is its effectiveness on relatively small, well-separated datasets and
its ability to model non-linear boundaries via kernel tricks like RBF. A notable limitation is that SVM does
not scale well to very large datasets and requires careful tuning of hyperparameters (C, gamma) and feature
scaling to perform well, making it less convenient than tree-based models on messy, large-scale real-world data.

## Files in this Repository
- `Assignment-6.ipynb` — full notebook (data collection, preprocessing, model, evaluation)
- `sample_weather_data.json` — bundled fallback dataset (used only if the live API is unreachable)
- `README.md` — this file
