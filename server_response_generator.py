# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

response_during_morning = 150
response_during_day = 200
response_during_evening = 170
response_during_night = 120

dates_list = []
server_response_list = []

current_date = pd.to_datetime('2024-05-01')

days_count = 10
for day_i in range(days_count):
    for hour_i in range(24):
        if 8 <= hour_i <= 19:
            server_response_list.append(round(np.random.normal(loc=response_during_day, scale=20, size=1).mean()))
        elif 5 <= hour_i <= 8:
            server_response_list.append(round(np.random.normal(loc=response_during_morning, scale=20, size=1).mean()))
        elif 19 <= hour_i <= 22:
            server_response_list.append(round(np.random.normal(loc=response_during_evening, scale=20, size=1).mean()))
        else:
            server_response_list.append(round(np.random.normal(loc=response_during_night, scale=20, size=1).mean()))

        dates_list.append(current_date)
        current_date = current_date + timedelta(hours=1)

server_response_df = pd.DataFrame({'Час': dates_list, 'Время отклика сервера': server_response_list})

server_response_df.set_index('Час', inplace=True)
server_response_df.index = pd.to_datetime(server_response_df.index)
print(server_response_df.head())

# Построим график времени отклика
plt.figure(figsize=(12, 6))
plt.plot(server_response_df.index, server_response_df['Время отклика сервера'], linestyle='-')
plt.title('Время отклика сервера по часам')
plt.xlabel('Час')
plt.ylabel('Время отклика сервера')
plt.grid(True)
plt.show()

server_response_df.to_csv('datasets/serverresponsetime.csv')
