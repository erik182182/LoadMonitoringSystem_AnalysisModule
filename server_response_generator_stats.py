# импортируем необходимые библиотеки
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime

dates_list = []
server_response_list = []

current_date = datetime.now().replace(microsecond=0)

times_count = 100
for i in range(times_count):
    server_response_list.append(round(np.random.normal(loc=100, scale=10, size=1).mean()))
    dates_list.append(current_date)
    current_date = current_date + timedelta(seconds=1)

server_response_df = pd.DataFrame({'Время': dates_list, 'Время отклика сервера': server_response_list})

server_response_df.set_index('Время', inplace=True)
server_response_df.index = pd.to_datetime(server_response_df.index)
print(server_response_df.head())

# Построим график времени отклика
plt.figure(figsize=(12, 6))
plt.plot(server_response_df.index, server_response_df['Время отклика сервера'], linestyle='-')
plt.title('Время отклика сервера по времени')
plt.xlabel('Время')
plt.ylabel('Время отклика сервера')
plt.grid(True)
plt.show()

server_response_df.to_csv('datasets/serverresponsetime_stats.csv')
