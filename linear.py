from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5], [6]]
y = [40, 54, 58, 69, 70, 90]
model = LinearRegression()
model.fit(X, y)
result = model.predict([[10]])
print(result)
