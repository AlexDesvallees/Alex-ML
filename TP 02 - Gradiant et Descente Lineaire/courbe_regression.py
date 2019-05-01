import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


def main():

    # Ajouts des coordonnées de 3 points (x,y)
    x=[[1], [8], [14]]
    y=[[1], [4], [7]]

    # On initialise la fonction de regression linéaire avec nos points
    linearRegressor = LinearRegression()
    linearRegressor.fit(x, y)

    # Affichage de la courbe qui définit notre régression linéaire
    plt.subplot(211)
    plt.scatter(x, y, color = 'red')
    plt.plot(x, linearRegressor.predict(x), color = 'blue')

    # Affichage des résidus liés à notre courbe
    plt.subplot(212)
    plt.scatter(x, y - linearRegressor.predict(x), color = 'red')
    plt.axhline(0, color='blue')

    # Affichage de la courbe
    plt.show()

if __name__ == "__main__":
	main()