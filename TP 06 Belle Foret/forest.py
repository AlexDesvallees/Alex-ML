from sklearn.ensemble import RandomForestClassifier

def main():
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X, Y)
    print(clf.predict([[.6, .6]]))

if __name__ == '__main__':
    main()