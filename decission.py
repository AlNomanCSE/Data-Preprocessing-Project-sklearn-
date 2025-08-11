from sklearn.tree import DecisionTreeClassifier

X = [[7, 2], [8, 3], [9, 8], [10, 9]]
# 1= apple 0=orange
y = [1, 1, 0, 0]

model = DecisionTreeClassifier()
model.fit(X,y)

result = model.predict([[5,6]])[0]
print(result)