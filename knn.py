from sklearn.neighbors import KNeighborsClassifier

X = [[180, 7], [200, 7.5], [250, 8], [300, 8.5], [330, 9], [360, 9.5]]
# 1= apple 0=orange
y = [0, 0, 0, 1, 1, 1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)
result = model.predict([[290,10]])[0]
print(result)