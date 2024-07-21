import pandas as pd

!wget -O covid19_india.csv https://raw.githubusercontent.com/amankharwal/Website-data/master/COVID19%20data%20for%20overall%20INDIA.csv
df = pd.read_csv('covid19_india.csv')
df.head()

import numpy as np
data = pd.read_csv("covid19_india.csv")
print(data.head()) 

df.sample(10)

data = data.drop("Date", axis=1)

import plotly.express as px
fig = px.bar(data, x='Date_YMD', y='Daily Confirmed')
fig.show()

cases = data["Daily Confirmed"].sum()
deceased = data["Daily Deceased"].sum()

labels = ["Confirmed", "Deceased"]
values = [cases, deceased]

fig = px.pie(data, values=values,
             names=labels,
             title='Daily Confirmed Cases vs Daily Deaths', hole=0.5)
fig.show()

death_rate = (data["Daily Deceased"].sum() / data["Daily Confirmed"].sum()) * 100
print(death_rate)

import plotly.express as px
fig = px.bar(data, x='Date_YMD', y='Daily Deceased')
fig.show()

pip install autots

from autots import AutoTS
model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(data, date_col="Date_YMD", value_col='Daily Deceased', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)
