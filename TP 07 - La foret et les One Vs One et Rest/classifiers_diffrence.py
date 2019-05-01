# -*- coding: latin-1 -*-

# --------------------------------------------
# --------------------------------------------
# ------- Code repris du calcul OvO ----------
# ------- et R de Nicolas Gourerau -----------
# --------------------------------------------
# --------------------------------------------


import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import itertools
import operator

#Awfull Hack to mute warning about 
# convergence issues
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# --------------------------------------------
# --------------- One Vs Rest ----------------
# --------------------------------------------
def generateOvRClassifier(classes, x_train, y_train):
    o_vs_r_classifiers = {}
    for elem in classes:
        class_valid = [x_train[index] for index, value in enumerate(y_train) if value == elem]
        class_invalid = [x_train[index] for index, value in enumerate(y_train) if value != elem]
        value = [1] * len(class_valid) + [0] * len(class_invalid)
        learn = class_valid + class_invalid
        o_vs_r_classifiers["%d_rest" % elem] = LogisticRegression(multi_class='ovr',solver='lbfgs').fit(learn, value)
    return o_vs_r_classifiers


def predictOVR(test_values, o_vs_r_classifiers):
    results = {}
    i=0
    for elem in test_values:
        intern_result = {}
        for name, classifier in o_vs_r_classifiers.items():
            result = classifier.predict([elem[0]])
            result_proba = classifier.predict_proba([elem[0]])
            intern_result[name.split('_')[0]] = result_proba[0][1]
        results[i] = intern_result
        i+=1
    correct = 0
    for key, elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct +=1
        #print("Predicted %s and value was %s" %(predicted,value))
    prct = (correct/len(results)*100)
    print(prct)
    # print(f"The One versus Rest score a {prct} % precision score")
    print("The One versus Rest score a {} % precision score".format(prct))

# --------------------------------------------
# ---------------- One Vs One ----------------
# --------------------------------------------

def generateOvOClassifier(classes, x_train, y_train):
    o_vs_o_classifiers = {}
    for elem in itertools.combinations(classes,2):
        class0 = [x_train[index] for index, value in enumerate(y_train) if value == elem[0]]
        class1 = [x_train[index] for index, value in enumerate(y_train) if value == elem[1]]
        value = [0] * len(class0) + [1] * len(class1)
        learn = class0 + class1
        o_vs_o_classifiers['%d_%d'%elem] = LogisticRegression(solver='lbfgs').fit(learn, value)
    return o_vs_o_classifiers

def predictOVO(test_values, o_vs_o_classifiers):
    """
    TO DO : STATS
    """
    results = {}
    i=0
    for elem in test_values:
        intern_result = {}
        for name,classifiers in o_vs_o_classifiers.items():
            result = classifiers.predict([elem[0]])
            members = name.split('_')
            if intern_result.get(members[result[0]]):
                intern_result[members[result[0]]] += 1
            else:
                intern_result[members[result[0]]] = 1
        results[i] = intern_result
        i+=1
    correct = 0
    for key,elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct += 1
        #print("Predicted %s and value was %s" %(predicted,value))
    prct = (correct/len(results)*100)
    # print(f"The One versus One score a {prct} % precision score ")
    print("The One versus One score a {} % precision score ".format(prct))

# --------------------------------------------
# ------------------ Forest ------------------
# --------------------------------------------

def generateForetClassifier(classes, x_train, y_train):
    foret_classifiers = {}
    for elem in classes:
        class_valid = [x_train[index] for index, value in enumerate(y_train) if value == elem]
        class_invalid = [x_train[index] for index, value in enumerate(y_train) if value != elem]
        value = [1] * len(class_valid) + [0] * len(class_invalid)
        learn = class_valid + class_invalid
        foret_classifiers["%d_rest" % elem] = RandomForestClassifier(n_estimators=10).fit(learn, value)
    return foret_classifiers

def predictForet(test_values, foret_classifiers):
    results = {}
    i=0
    for elem in test_values:
        intern_result = {}
        for name, classifier in foret_classifiers.items():
            result = classifier.predict([elem[0]])
            result_proba = classifier.predict_proba([elem[0]])
            intern_result[name.split('_')[0]] = result_proba[0][1]
        results[i] = intern_result
        i+=1
    correct = 0
    for key, elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct +=1
    prct = (correct/len(results)*100)
    # print(f"The Forest score a {prct} % precision score ")
    print("The Forest score a {} % precision score ".format(prct))

def main():
    # On récupère notre jeux de données (jdd) digits
    digits = datasets.load_digits()
    data = digits['data']
    target  = digits['target']
    classes = set(target)
    # On va split notre jdd en test et train
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    # Création de tuples images / valeurs
    test_values = [(x_test[index],value) for index, value in enumerate(y_test)]

    # Création des classifieurs One vs One
    o_vs_o_classifiers = generateOvOClassifier(classes, x_train, y_train)
    # Avec la prédiction
    predictOVO(test_values, o_vs_o_classifiers)
    
    # Création des classifieurs One vs Rest (ou All)
    ovrclassifier = generateOvRClassifier(classes, x_train, y_train)
    # Avec sa prédiction
    predictOVR(test_values,ovrclassifier)

    # Création des classifieurs Forest
    foret_classifiers = generateForetClassifier(classes, x_train, y_train)
    # Avec leur prédictions
    predictForet(test_values, foret_classifiers)

if __name__ == "__main__":
    main()