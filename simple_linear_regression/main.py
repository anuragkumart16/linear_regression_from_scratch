def calculate_mse(predicted_array, output_array):
    mse = 0

    for index in range(len(predicted_array)):
        mse += (predicted_array[index] - output_array[index]) ** 2

    mse /= len(predicted_array)

    return mse

class LinearRegression:
    def __init__(self):
        self.weight = 0
        self.bias = 0

    def get_attributes(self):
        return {"weight": self.weight, "bias": self.bias}

    def calculate_delta_parameters(self, input_array, predicted_array, actual_array):
        delta_weight = 0
        for index in range(len(input_array)):
            delta_weight += (
                predicted_array[index] - actual_array[index]
            ) * input_array[index]

        delta_weight = (delta_weight * 2) / len(input_array)

        delta_bias = 0
        for index in range(len(input_array)):
            delta_bias += predicted_array[index] - actual_array[index]
        delta_bias = (delta_bias * 2) / len(input_array)

        return (delta_weight, delta_bias)

    def train(self, input_array, output_array, learning_rate):
        predicted_array = []
        for element in input_array:
            predicted_array.append(self.weight * element + self.bias)

        delta_weight, delta_bias = self.calculate_delta_parameters(
            input_array, predicted_array, output_array
        )

        self.weight -= learning_rate * delta_weight
        self.bias -= learning_rate * delta_bias
        return (self.get_attributes(),calculate_mse(predicted_array,output_array))

    def predict(self, x):
        return self.weight * x + self.bias
