class MultipleLinearRegression:
    def __init__(self):
        self.weights = []
        self.bias = 0

    def get_attribute(self):
        return (self.weights,self.bias)

    def init_weights(self,X):
        for element in range(len(X)):
            self.weights.append(0)

    def calculate_weighted_sum(self,X):
        weighted_sum = 0
        for index in range(len(X)):
            weighted_sum += X[index]*self.weights[index]
        weighted_sum += self.bias
        return weighted_sum


    def train(self,X,y,learning_rate):
        weighted_sum  = self.calculate_weighted_sum(X)
        error = weighted_sum - y 
        change_in_weights = []

        for x in X:
            gradient = 2 * error * x
            change_in_weights.append(gradient) 

        # update weights 
        for index in range(len(change_in_weights)):
            self.weights[index] -= learning_rate * change_in_weights[index]

        self.bias -= learning_rate * 2 * error

        return self.weights,self.bias

    def predict(self,X):
        weighted_sum = 0
        for x,weights in zip(X,self.weights):
            weighted_sum += x*weights

        weighted_sum += self.bias
        return weighted_sum

    

        









        



