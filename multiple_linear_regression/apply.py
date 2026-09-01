from main import MultipleLinearRegression


model = MultipleLinearRegression()


# ---- configuration ----
epochs = 1000000
learning_rate = 0.01


# ---- dataset ----
# y = 2x1 + 3x2 + 5

X = [
    [1, 1],
    [2, 1],
    [1, 2],
    [3, 2],
    [2, 3],
    [4, 3]
]

y = [10, 12, 13, 17, 18, 22]


# ---- training ----
model.init_weights(X[0])

for epoch in range(epochs):

    weights, bias = model.train(X, y, learning_rate)

    mse = model.calculate_mse(X, y)

    print(
        f"Epoch {epoch + 1}: "
        f"Weights = {weights}, "
        f"Bias = {bias}, "
        f"MSE = {mse}"
    )


# ---- predicting ----
x_predict = [0, 0]

print(
    model.predict(x_predict),
    "is our response for",
    x_predict
)