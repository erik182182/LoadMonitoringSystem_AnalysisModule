# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# сразу превратим дату в индекс и преобразуем ее в объект datetime
births = pd.read_csv('../datasets/births.csv', index_col='Date', parse_dates=True)
print(births.head(3))

# импортируем функцию seasonal_decompose из statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose

# задаем размер графика
from pylab import rcParams

rcParams['figure.figsize'] = 11, 9

# применяем функцию к данным о перевозках
decompose = seasonal_decompose(births)
decompose.plot()

plt.show()

# импортируем необходимую функцию
from statsmodels.tsa.stattools import adfuller

# передадим ей столбец с данными о перевозках и поместим результат в adf_test
adf_test = adfuller(births['Births'])

# выведем p-value
print('p-value = ' + str(adf_test[1]))

# импортируем автокорреляционную функцию (ACF)
from statsmodels.graphics.tsaplots import plot_acf

# применим ее к данным о пассажирах
plot_acf(births)
plt.axis('tight')
plt.show()
