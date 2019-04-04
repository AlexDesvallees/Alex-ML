import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

x=[["2019-03-16"], ["2019-03-18"], ["2019-03-20"], ["2019-03-24"], ["2019-03-25"], ["2019-03-26"], ["2019-03-27"], ["2019-03-28"], ["2019-03-30"], ["2019-04-03"]]
# convX = x.astype(np.float64)
y=[[81682], [81720], [81760], [81826], [81844], [81864], [81881], [81900], [81933], [82003]]

linearRegressor = LinearRegression()
reg= linearRegressor.fit(x, y)

plt.subplot(211)
plt.scatter(x, y, color = 'red')
plt.plot(x, linearRegressor.predict(x), color = 'blue')

plt.subplot(212)
plt.scatter(x, y - linearRegressor.predict(x), color = 'red')
plt.axhline(0, color='blue')

plt.show()

# Exemple de regression lineaire repris de ce que l'on avait fait
# Convertion des dates en valeures flotantes en cours, pas de compteurs