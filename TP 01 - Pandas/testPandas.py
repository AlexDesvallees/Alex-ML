# from pandas import DataFrame, read_csv
# import matplotlib.pyplot as plt
# import pandas as pd 
 
# file = r'fbi_ucr.xls'
# df = pd.read_excel(file)
# print(df.head())

import plotly as pt
# import plotly.plotly as py
# import plotly.graph_objs as go
pt.tools.set_credentials_file(username='Alexode', api_key='3E82qm2qQlaqQ4A48a0q')

import numpy as np
import pandas as pd

# N = 400
# x = np.linspace(0,1,N)
# y = np.random.randn(N)
# df = pd.DataFrame({'x': x, 'y': y})
# file = r'Popular_Baby_Names.csv'
# df = pd.read_csv(file)
# df.sort_values(by='Count', ascending=False).head()

# data = [
#     go.Bar(
#         x=df['Child\'s First Name'],
#         y=df['Count']
#         # marker={
#         #     'color': ys,
#         #     'colorscale': 'Viridis'
#         # }
#     )
# ]



# url = py.plot(data, filename='pandas-bar-chart')

# print(df.head())
# print(df.describe())
# print(df.sort_values(by='Count', ascending=False).head())

import plotly.plotly as py
import plotly.graph_objs as go

file = r'Popular_Baby_Names.csv'
df = pd.read_csv(file)
sorted_df = df.sort_values(by='Count', ascending=False).head()
print(sorted_df)

xs = sorted_df['Child\'s First Name']
ys = sorted_df['Count']
data = [go.Bar(
    x=xs,
    y=ys,
    marker={
        'color': ys,
        'colorscale': 'Viridis'
    }
)]

# layout = {
#     'xaxis': {
#         'categoryorder': 'array',
#         'categoryarray': [x for _, x in sorted(zip(xs, ys))]
#     }
# }

layout = {
    'xaxis': {
        'categoryorder': 'category ascending'
    },
    'yaxis': {
        'categoryorder': 'category ascending'
        # 'categoryarray': [x for _, x in sorted(zip(xs, ys))]
    }
}

fig = go.FigureWidget(data=data, layout=layout)
url = py.plot(fig, filename='pandas-bar-chart-2')