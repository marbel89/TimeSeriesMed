# Time Series Analysis and Forecasting Project

This project focuses on time series analysis and forecasting using statistical learning techniques. In this case it is SARIMA. Please refer to the individual Notebook file for further discussion.

## Tasks

1. Forecast monthly number of complaints using precipitation
2. Modeling and forecasting seasonal patterns
3. Predicting and forecasting sunspots counts

## Code Snippets

### Data Preprocessing and Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load and preprocess data
df = pd.read_csv('your_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Visualize time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'])
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Seasonal decomposition
result = seasonal_decompose(df['value'], model='additive')
result.plot()
plt.show()

# ACF and PACF plots
plot_acf(df['value'])
plot_pacf(df['value'])
plt.show()
```

### Model Training and Forecasting

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Split data into train and test sets
train = df[:int(0.8*len(df))]
test = df[int(0.8*len(df)):]

# Fit ARIMA model
model = ARIMA(train['value'], order=(1,1,1))
results = model.fit()

# Make predictions
predictions = results.forecast(steps=len(test))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test['value'], predictions))
print(f'RMSE: {rmse}')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['value'], label='Train')
plt.plot(test.index, test['value'], label='Test')
plt.plot(test.index, predictions, label='Predictions')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()
```

## Results

The project includes various visualizations and analyses:

1. Time series plots showing the original data and forecasts
2. Seasonal decomposition plots
3. ACF and PACF plots for model selection
4. Forecast accuracy metrics (e.g., RMSE)

## Future Work

1. Explore more advanced forecasting methods (e.g., Prophet, LSTM)
2. Incorporate exogenous variables for improved predictions
3. Implement cross-validation for more robust model evaluation

## Dependencies

- pandas
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- numpy

## Usage

1. Clone the repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebooks or Python scripts to reproduce the analysis
