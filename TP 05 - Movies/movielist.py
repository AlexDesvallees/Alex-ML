from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

# Charge movielens-100k dataset
movielens_ds = Dataset.load_builtin('ml-100k')

# Creer un jeu de test et de train ( 15%, 85%)
trainset, testset = train_test_split(movielens_ds, test_size=.15)

algo = KNNWithMeans()

# Train sur le jeu de donnée trainset
algo.fit(trainset)
# Prediction sur le jeu de donnée testset
predictions = algo.test(testset)

# Affiche le RMSE
accuracy.rmse(predictions)

#print(predictions)

result =[]
for prediction in predictions:
    # Difference prediction et realite
    result.append(prediction.r_ui - prediction.est)

# Histogramme du resultat
plt.hist(result, 100)

plt.show()