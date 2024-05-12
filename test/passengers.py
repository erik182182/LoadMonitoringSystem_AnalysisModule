# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# импортируем файл с данными о пассажирах
passengers = pd.read_csv("../datasets/passengers.csv")
# превратим дату в индекс и сделаем изменение постоянным
passengers.set_index('Month', inplace=True)
print(passengers.head())

# превратим дату (наш индекс) в объект datetime
passengers.index = pd.to_datetime(passengers.index)

# посмотрим на первые пять дат и на тип данных
print(passengers['1949-05-02':'1950-03'])

# зададим размер графика
plt.figure(figsize=(15, 8))

# поочередно зададим кривые (перевозки и скользящее среднее) с подписями и цветом
plt.plot(passengers, label='Перевозки пассажиров по месяцам', color='steelblue')
plt.plot(passengers.rolling(window=12).mean(), label='Скользящее среднее за 12 месяцев', color='orange')

# добавим легенду, ее положение на графике и размер шрифта
plt.legend(title='', loc='upper left', fontsize=14)

# добавим подписи к осям и заголовки
plt.xlabel('Месяцы', fontsize=14)
plt.ylabel('Количество пассажиров', fontsize=14)
plt.title('Перевозки пассажиров с 1949 по 1960 год', fontsize=16)

# выведем обе кривые на одном графике
plt.show()

# импортируем функцию seasonal_decompose из statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose

# задаем размер графика
from pylab import rcParams

rcParams['figure.figsize'] = 11, 9

# применяем функцию к данным о перевозках
decompose = seasonal_decompose(passengers)
decompose.plot()

plt.show()

# импортируем необходимую функцию
from statsmodels.tsa.stattools import adfuller

# передадим ей столбец с данными о перевозках и поместим результат в adf_test
adf_test = adfuller(passengers['#Passengers'])

# выведем p-value
print('p-value = ' + str(adf_test[1]))

# импортируем автокорреляционную функцию (ACF)
from statsmodels.graphics.tsaplots import plot_acf

# применим ее к данным о пассажирах
plot_acf(passengers)
plt.axis('tight')
plt.show()

# обучающая выборка будет включать данные до декабря 1959 года включительно
train = passengers[:'1959-12']

# тестовая выборка начнется с января 1960 года (по сути, один год)
test = passengers['1960-01':]

plt.plot(train, color="black")
plt.plot(test, color="red")

# заголовок и подписи к осям
plt.title('Разделение данных о перевозках на обучающую и тестовую выборки')
plt.ylabel('Количество пассажиров')
plt.xlabel('Месяцы')

# добавим сетку
plt.grid()

plt.show()

# принудительно отключим предупреждения системы
import warnings

warnings.simplefilter(action='ignore', category=Warning)

# обучим модель с соответствующими параметрами, SARIMAX(3, 0, 0)x(0, 1, 0, 12)
# импортируем класс модели
from statsmodels.tsa.statespace.sarimax import SARIMAX

# создадим объект этой модели
model = SARIMAX(train,
                order=(3, 0, 0),
                seasonal_order=(0, 1, 0, 12))

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
plt.plot(train, color="black")
plt.plot(test, color="red")
plt.plot(predictions, color="green")

# заголовок и подписи к осям
plt.title("Обучающая выборка, тестовая выборка и тестовый прогноз")
plt.ylabel('Количество пассажиров')
plt.xlabel('Месяцы')

# добавим сетку
plt.grid()

plt.show()

# импортируем метрику
from sklearn.metrics import mean_squared_error

# рассчитаем MSE
print(mean_squared_error(test, predictions))

# и RMSE
print(np.sqrt(mean_squared_error(test, predictions)))

# прогнозный период с конца имеющихся данных
start = len(passengers)

# и закончится 36 месяцев спустя
end = (len(passengers) - 1) + 3 * 12

# теперь построим прогноз на три года вперед
forecast = result.predict(start, end)

# посмотрим на весь 1963 год
print(forecast[-12:])

# выведем две кривые (фактические данные и прогноз на будущее)
plt.plot(passengers, color='black')
plt.plot(forecast, color='blue')

# заголовок и подписи к осям
plt.title('Фактические данные и прогноз на будущее')
plt.ylabel('Количество пассажиров')
plt.xlabel('Месяцы')

# добавим сетку
plt.grid()

plt.show()

