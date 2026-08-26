from main import LinearRegression

model = LinearRegression()

# ---- Train Model ------
# dataset
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# configuration
epoches = 100
learning_rate = 0.01

# running epoches 
for epoch in range(epoches):
    attributes, mse = model.train(x,y,learning_rate)
    print(f"for epoch {epoch} the weight is {attributes['weight']} and bias is {attributes['bias']} for mse : {mse}")

# getting final weight and bias
final_attributes = model.get_attributes()
final_weight = final_attributes['weight']
final_bias = final_attributes['bias']

# ---- Make Prediction ----
x = 0
predicted_value =  model.predict(x)
print(f" model predicted {predicted_value} for input {x}")








