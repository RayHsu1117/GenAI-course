import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('./data/467410-2022-08.csv')
# Calculate average wind speed for each wind direction interval
df.loc[1:,'風向(360degree)'] = df.loc[1:,'風向(360degree)'].astype(int)
df.loc[1:,'風速(m/s)'] = df.loc[1:,'風速(m/s)'].astype(float)
wind_direction_intervals = [0, 90, 180, 270, 360]
average_wind_speeds = []
for i in range(len(wind_direction_intervals) - 1):
    lower_bound = wind_direction_intervals[i]
    upper_bound = wind_direction_intervals[i + 1]
    interval_data = []  # Initialize interval_data
    for j in range(1,len(df)):
        if lower_bound <= df.loc[j,'風向(360degree)'] <= upper_bound:
            interval_data.append(df.loc[j,'風速(m/s)'])
    average_wind_speed = sum(interval_data) / len(interval_data)
    #average_wind_speed = interval_data['風速(m/s)'].mean()
    average_wind_speeds.append(average_wind_speed)

# Plot radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(wind_direction_intervals)-1, endpoint=False).tolist()
angles += angles[:1]
# Display average wind speed for each wind direction interval

for i in range(len(wind_direction_intervals) - 1):
    #angle = (wind_direction_intervals[i] + wind_direction_intervals[i + 1]) / 2
    angle = angles[i]+0.03
    speed = average_wind_speeds[i]
    ax.text(angle, speed, f'{speed:.2f} m/s', ha='center', va='top')
average_wind_speeds += average_wind_speeds[:1]
ax.plot(angles, average_wind_speeds, color='b', linewidth=1)
ax.fill(angles, average_wind_speeds, color='b', alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['0-90°', '90-180°', '180-270°', '270-360°'])
ax.set_yticklabels([])
ax.set_title('Average Wind Speed by Wind Direction')
ax.grid(True)

# Show the radar chart
plt.show()


# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('date(day)')
# ax1.set_ylabel('temparature(℃)', color=color)
# #ax1.plot(df['觀測時間(day)'][1:], df['氣溫(℃)'][1:].astype(float), marker='o', color=color)  # 设置红色实心圆点的折线
# ax1.plot(df.loc[1:,'觀測時間(day)'],df.loc[1:,'氣溫(℃)'].astype(float), marker='o', color=color)  # 设置红色实心圆点的折线
# ax1.tick_params(axis='y', labelcolor=color)
# #ax1.legend(loc='upper left')  # 添加图例
# # Plot temperature
# #plt.plot(df['觀測時間(day)'][1:], df['氣溫(℃)'][1:], label='Temperature')
# ax2 = ax1.twinx()  

# # 绘制第二组数据
# color = 'tab:blue'
# ax2.set_ylabel('Rainfall(mm)', color=color)
# df.loc[1:,'降水量(mm)'] = df.loc[1:,'降水量(mm)'].replace('T', 0.0)
# ax2.plot(df.loc[1:,'觀測時間(day)'], df.loc[1:,'降水量(mm)'].astype(float), marker='s', color=color, linestyle='--')  # 设置蓝色虚线的折线
# ax2.tick_params(axis='y', labelcolor=color)
# #ax2.legend(loc='upper right')  # 添加图例

# fig.tight_layout()  # 调整子图之间的间距
# # Plot rainfall
# #plt.plot(df['觀測時間(day)'][1:], df['降水量(mm)'][1:], label='Rainfall')

# # Set labels and title
# #
# # Add legend
# #plt.legend()

# # Show the plot
# plt.show()