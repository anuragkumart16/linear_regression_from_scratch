class MultipleLinearRegression:
    def __init__(self):
        self.weights = []
        self.bias = 0

    def get_attribute(self):
        return self.weights, self.bias

    def init_weights(self, X):
        for _ in range(len(X)):
            self.weights.append(0)

    def calculate_weighted_sum(self, X):
        weighted_sum = 0

        for index in range(len(X)):
            weighted_sum += X[index] * self.weights[index]

        weighted_sum += self.bias

        return weighted_sum

    def train(self, X, y, learning_rate):

        number_of_samples = len(X)

        # --------------------------------
        # 1. Calculate predictions
        # --------------------------------

        predictions = []

        for x in X:
            prediction = self.calculate_weighted_sum(x)
            predictions.append(prediction)

        # --------------------------------
        # 2. Calculate errors
        # --------------------------------

        errors = []

        for prediction, actual in zip(predictions, y):
            error = prediction - actual
            errors.append(error)

        # --------------------------------
        # 3. Calculate gradients
        # --------------------------------

        change_in_weights = []

        for feature_index in range(len(self.weights)):
            gradient = 0

            for sample_index in range(number_of_samples):
                gradient += errors[sample_index] * X[sample_index][feature_index]

            gradient *= 2
            gradient /= number_of_samples

            change_in_weights.append(gradient)

        # --------------------------------
        # 4. Calculate bias gradient
        # --------------------------------

        bias_gradient = 0

        for error in errors:
            bias_gradient += error

        bias_gradient *= 2
        bias_gradient /= number_of_samples

        # --------------------------------
        # 5. Update weights
        # --------------------------------

        for index in range(len(self.weights)):
            self.weights[index] -= learning_rate * change_in_weights[index]

        # --------------------------------
        # 6. Update bias
        # --------------------------------

        self.bias -= learning_rate * bias_gradient

        return self.weights, self.bias

    def predict(self, X):

        weighted_sum = 0

        for x, weight in zip(X, self.weights):
            weighted_sum += x * weight

        weighted_sum += self.bias

        return weighted_sum
