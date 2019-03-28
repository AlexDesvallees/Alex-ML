from PIL import Image
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# On stock les donnees de digits dans une var dataset
digits_ds = datasets.load_digits()

# Si on veut plus de details
# print(digits_ds.keys())
# print(digits_ds.data)

# Initialisation des variables
compteur = len(digits_ds.data)
myList = []
myRes = []

myRes_X = []
myRes_Y = []

# On recupere les index contenant les 0 et 1
while compteur > 0 :
    if(digits_ds.target[compteur-1]==0 or digits_ds.target[compteur-1]==1) :
        myList.append(compteur-1)
    compteur -= 1

# print(myList)

# Puis on les associes avec les data dans une liste
for i in myList :
    myRes.append([digits_ds.data[i], digits_ds.target[i]])
    myRes_X.append(digits_ds.data[i])
    myRes_Y.append(digits_ds.target[i])
# print(myRes)

# Pour chaques classes on affiche le nombre present dans notre data
for i in [0,1] :
    print("classe : %s, nb occurences : %s" % (i, len(digits_ds.target[digits_ds.target == i])))

# Test sur le nombre total de donnees dans notre liste de 0 et 1
#print(len(myRes))

# Separation en listes de test et de train
x_train, x_test, y_train, y_test = train_test_split(myRes_X, myRes_Y, test_size = 0.33, random_state = 40)
#print(len(myRes) - len(myRes_test))
#print(len(myRes_train))

class0 = [x_train[index] for index, value in enumerate(y_train) if value == 0]

class1 = [x_train[index] for index, value in enumerate(y_train) if value == 1]

value = [0] * len(class0) + [1] * len(class1)
learn = class0 + class1
o_vs_o_classifiers = LogisticRegression(solver='lbfgs').fit(learn, value)

test_values = [(x_test[index],value) for index, value in enumerate(y_test)]

for elem in test_values:
    result = o_vs_o_classifiers.predict([elem[0]])
    print("Resultat : ", result)
    print("Attendu :", elem[1])
    if (elem[1]==result):
        print("OK")
    else:
        print("NOK")

# def generateOvRClassifier(classes):
#     o_vs_r_classifiers = {}
#     for elem in classes:
#         class_valid = [x_train[index] for index, value in enumerate(y_train) if value == elem]
#         class_invalid = [x_train[index] for index, value in enumerate(y_train) if value != elem]
#         value = [1] * len(class_valid) + [0] * len(class_invalid)
#         learn = class_valid + class_invalid
#         o_vs_r_classifiers["%d_rest" % elem] = LogisticRegression(multi_class='ovr',solver='lbfgs').fit(learn, value)
#     return o_vs_r_classifiers


# def predictOVR(test_values, o_vs_r_classifiers):
#     results = {}
#     i=0
#     for elem in test_values:
#         intern_result = {}
#         for name, classifier in o_vs_r_classifiers.items():
#             result = classifier.predict([elem[0]])
#             result_proba = classifier.predict_proba([elem[0]])
#             intern_result[name.split('_')[0]] = result_proba[0][1]
#         results[i] = intern_result
#         i+=1
#     correct = 0
#     for key, elem in results.items():
#         predicted = max(elem.items(), key=operator.itemgetter(1))[0]
#         value = test_values[key][1]
#         if int(predicted) == value:
#             correct +=1
#         #print("Predicted %s and value was %s" %(predicted,value))
#     prct = (correct/len(results)*100)
#     print(f"The One versus Rest score a {prct} % precision score ")

# # One vs Rest/All
# OvAClassifier = generateOvRClassifier(set(digits_ds['target']))
# predictOVR(test_values, OvAClassifier)