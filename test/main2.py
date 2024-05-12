import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Создаем даты с января 2020 года по декабрь 2022 года
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')

# Генерируем случайные значения температуры
temperature = np.random.normal(loc=25, scale=5, size=len(dates))

# Создаем DataFrame
temperature_df = pd.DataFrame({'Дата': dates, 'Температура': temperature})

# Устанавливаем 'Дата' в качестве индекса
temperature_df.set_index('Дата', inplace=True)

# Выводим первые несколько строк
print(temperature_df.head())

# Построим график температуры
plt.figure(figsize=(12, 6))
plt.plot(temperature_df.index, temperature_df['Температура'], linestyle='-')
plt.title('Средняя дневная температура')
plt.xlabel('Дата')
plt.ylabel('Температура (°C)')
plt.grid(True)
plt.show()

# Обучение модели SARIMA
model = SARIMAX(temperature_df['Температура'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
model_fit = model.fit()

# Вывод статистики модели
print(model_fit.summary())

# Прогноз на основе обученной модели
forecast = model_fit.forecast(steps=14)

# Рассчитываем MSE и MAE
mse = mean_squared_error(temperature_df['Температура'][-14:], forecast)
mae = mean_absolute_error(temperature_df['Температура'][-14:], forecast)

print(f'MSE: {mse}')
print(f'MAE: {mae}')

# Прогноз на будущее (следующие 7 дней)
forecast_future = model_fit.get_forecast(steps=7)


# Создаем новый DataFrame для будущих значений
future_dates = pd.date_range(start='2022-12-31', periods=7, freq='D') + pd.DateOffset(days=1)
forecast_df = pd.DataFrame({'Дата': future_dates, 'Прогноз температуры': forecast_future.predicted_mean})


# Присоединяем прогноз к исходному DataFrame
temperature_df = temperature_df._append(forecast_df)


# Визуализация исходных данных и прогноза
plt.figure(figsize=(12, 6))
plt.plot(temperature_df.index[:-7], temperature_df['Температура'][:-7], label='Исходные данные')
plt.plot(temperature_df.index[-7:], temperature_df['Прогноз температуры'][-7:], label='Прогноз')
plt.title('Прогноз средней дневной температуры')
plt.xlabel('Дата')
plt.ylabel('Температура (°C)')
plt.legend()
plt.grid(True)
plt.show()
