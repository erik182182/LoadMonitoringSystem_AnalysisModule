# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# импортируем файл с данными о времени отклика
server_response = pd.read_csv("datasets/serverresponsetime.csv")
server_response.set_index('Час', inplace=True)
server_response.index = pd.to_datetime(server_response.index)
print(server_response.head())

server_response_plot = server_response['2024-05-08':]

# Построим график времени отклика (За последние дни)
plt.figure(figsize=(12, 6))
plt.plot(server_response_plot.index, server_response_plot['Время отклика сервера'], linestyle='-')
plt.title('Время отклика сервера по часам')
plt.xlabel('Час')
plt.ylabel('Время отклика сервера')
plt.grid()
plt.show()

# импортируем функцию seasonal_decompose из statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose

# задаем размер графика
from pylab import rcParams

rcParams['figure.figsize'] = 11, 9

# применяем функцию к данным о времени отклика сервера (за последние дни)
decompose = seasonal_decompose(server_response_plot)
decompose.plot()

plt.show()

# импортируем автокорреляционную функцию (ACF)
from statsmodels.graphics.tsaplots import plot_acf

# применим функцию к нашему набору данных (за последние дни)
plot_acf(server_response_plot['Время отклика сервера'], lags=np.arange(len(server_response_plot['Время отклика сервера'])), alpha=None)

# добавим отступы сверху и снизу на графике
plt.axis('tight')
plt.show()


# Прогноз

# обучающая выборка
train = server_response[:'2024-05-07']

# тестовая выборка
test = server_response['2024-05-08':]

plt.plot(train, color="black")
plt.plot(test, color="red")

# заголовок и подписи к осям
plt.title('Разделение данных на обучающую и тестовую выборки')
plt.ylabel('Время отклика сервера')
plt.xlabel('Час')

# добавим сетку
plt.grid()

plt.show()

# # импортируем функцию для автоматического подбора параметров модели ARIMA
# from pmdarima import auto_arima
#
#
# # "погасим" предупреждения
# import warnings
# warnings.filterwarnings("ignore")
#
# # настроим поиск параметров на обучающей выборке
# parameter_search = auto_arima(train, start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 24, start_P = 0, seasonal = True,
#                          d = None, D = 1, trace = True, error_action ='ignore', suppress_warnings = True,  stepwise = True)           #
#
# # выведем результат
# print(parameter_search.summary())

# Best model:  ARIMA(1,0,1)(2,1,0)[24]

# принудительно отключим предупреждения системы
import warnings

warnings.simplefilter(action='ignore', category=Warning)

# обучим модель с соответствующими параметрами, (1,0,1)(2,1,0)[24]
# импортируем класс модели
from statsmodels.tsa.statespace.sarimax import SARIMAX

# создадим объект этой модели
model = SARIMAX(train,
                order=(1, 0, 1),
                seasonal_order=(2, 1, 0, 24))

# применим метод fit
result = model.fit()

# тестовый прогнозный период начнется с конца обучающего периода
start = len(train)

# и закончится в конце тестового
end = len(train) + len(test) - 1

# применим метод predict
predictions = result.predict(start, end)
print(predictions)

# выведем три кривые (обучающая, тестовая выборка и тестовый прогноз)
plt.plot(train[-24:], color="black", label='Обучающая выборка')
plt.plot(test, color="red", label='Тестовая выборка')
plt.plot(predictions, color="green", label='Прогноз')

# заголовок и подписи к осям
plt.title("Обучающая выборка, тестовая выборка и тестовый прогноз")
plt.ylabel('Время отклика сервера')
plt.xlabel('Час')

# добавим сетку
plt.grid()
plt.legend()

plt.show()

# импортируем метрику
from sklearn.metrics import mean_squared_error

# рассчитаем MSE
print('MSE: ')
print(mean_squared_error(test, predictions))

# и RMSE
print('RMSE: ')
print(np.sqrt(mean_squared_error(test, predictions)))

# прогнозный период с конца имеющихся данных
start = len(server_response)

# и закончится 1 день спустя
end = (len(server_response) - 1) + 1 * 24

# теперь построим прогноз на 1 день вперед
forecast = result.predict(start, end)

# выведем две кривые (фактические данные и прогноз на будущее)
plt.plot(server_response_plot, color='black', label='Фактические данные')
plt.plot(forecast, color='blue', label='Прогноз')

# заголовок и подписи к осям
plt.title('Фактические данные и прогноз на будущее')
plt.ylabel('Время отклика сервера')
plt.xlabel('Час')

# добавим сетку
plt.grid()
plt.legend()

plt.show()
