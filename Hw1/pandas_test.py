import pandas as pd
import numpy as np
#有用的row[0,16,17,21,22,31]
df = pd.read_csv('./data/467410-2022-08.csv')
df.insert(column='UVI level',value='0',loc=32)

#df['UVI level'][0] = 'UVI level'
df.loc[0,'UVI level'] = 'UVI level'
df.loc[1:,'日最高紫外線指數'] = df.loc[1:,'日最高紫外線指數'].astype(int)
#df['日最高紫外線指數'][1:] = df['日最高紫外線指數'][1:].astype(int)

conditions = [
    (df.loc[1:,'日最高紫外線指數'] <= 2),
    (df.loc[1:,'日最高紫外線指數'] <= 5),
    (df.loc[1:,'日最高紫外線指數'] <= 7),
    (df.loc[1:,'日最高紫外線指數'] <= 10)
]

choices = ['低', '中', '高', '甚高']

# 使用 np.select 來替換值
df.loc[1:,'UVI level'] = np.select(conditions, choices, default='極高')
uvi_counts = df.loc[1:,'UVI level'].value_counts()
print(uvi_counts)

df.loc[1:,'降水量(mm)'] = df.loc[1:,'降水量(mm)'].replace('T', 0.0)


#df['降水量(mm)'][1:] = df['降水量(mm)'][1:].astype(float)
df.loc[1:,'降水量(mm)'] = df.loc[1:,'降水量(mm)'].astype(float)
df.loc[1:,'降水時數(hour)'] = df.loc[1:,'降水時數(hour)'].astype(float)
df.insert(column='降水強度(mm/hour)',value='0.0',loc=23)

df.loc[0,'降水強度(mm/hour)'] = '降水強度(mm/hour)'
df.loc[1:,'降水強度(mm/hour)'] = df.loc[1:,'降水強度(mm/hour)'].astype(float)
for i in range(1,len(df)):
    if df.loc[i,'降水時數(hour)'] == 0.0:
        df.loc[i,'降水強度(mm/hour)'] = 0.0
    else:
        df.loc[i,'降水強度(mm/hour)'] = df.loc[i,'降水量(mm)'] / df.loc[i,'降水時數(hour)']
#df['降水強度'][1:] = np.where(df['降水時數(hour)'][1:] == 0.0, 0.0, df['降水量(mm)'][1:] / df['降水時數(hour)'][1:])

#df['降水強度'][1:] = df['降水量(mm)'][1:] / df['降水時數(hour)'][1:]
average_intensity = df.loc[1:,'降水強度(mm/hour)'].mean()
#print("Average Intensity:", average_intensity)
for i in range(1,len(df)):
       if df.loc[i,'降水強度(mm/hour)'] > average_intensity:
              print(df.iloc[i])

# df.loc[:,'降水強度(mm/hour)'].astype(str)

#df.to_csv('new_467410-2022-08.csv', index=False)

