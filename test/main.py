import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Создаем даты с января 2022 года по декабрь 2023 года
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='M')

# Генерируем случайные продажи в интервале от 1000 до 5000
sales = np.random.randint(1000, 5000, size=len(dates))

# Создаем DataFrame
sales_df = pd.DataFrame({'Дата': dates, 'Продажи': sales})

# Устанавливаем 'Дата' в качестве индекса
sales_df.set_index('Дата', inplace=True)

# Выводим
print(sales_df)

# Построим график продаж
plt.figure(figsize=(12, 6))
plt.plot(sales_df.index, sales_df['Продажи'], marker='o', linestyle='-')
plt.title('Продажи в магазине МВидео')
plt.xlabel('Дата')
plt.ylabel('Продажи')
plt.grid(True)
plt.show()

# Обучение модели ARIMA
model = ARIMA(sales_df['Продажи'], order=(1, 1, 1))
model_fit = model.fit()

# Вывод статистики модели
print(model_fit.summary())

# Прогноз на основе обученной модели
forecast = model_fit.forecast(steps=12)

# Рассчитываем MSE и MAE
mse = mean_squared_error(sales_df['Продажи'][-12:], forecast)
mae = mean_absolute_error(sales_df['Продажи'][-12:], forecast)

print(f'MSE: {mse}')
print(f'MAE: {mae}')

# Прогноз на будущее (следующие 12 месяцев)
forecast_future = model_fit.forecast(steps=12)

# Создаем новый DataFrame для будущих значений
future_dates = pd.date_range(start='2024-01-01', periods=12, freq='ME')
forecast_df = pd.DataFrame({'Дата': future_dates, 'Прогноз продаж': forecast_future})

# Присоединяем прогноз к исходному DataFrame
sales_df = sales_df._append(forecast_df, ignore_index=True)

# Визуализация исходных данных и прогноза
plt.figure(figsize=(12, 6))
plt.plot(sales_df.index[:-12], sales_df['Продажи'][:-12], label='Исходные данные')
plt.plot(sales_df.index[-12:], sales_df['Прогноз продаж'][-12:], label='Прогноз')
plt.title('Прогноз продаж в магазине МВидео')
plt.xlabel('Дата')
plt.ylabel('Продажи')
plt.legend()
plt.grid(True)
plt.show()
