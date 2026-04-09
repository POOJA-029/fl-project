import flwr as fl
from model import create_model
from utils import load_data

(X_train, X_test, y_train, y_test) = load_data("data/diabetes.csv")

model = create_model()

class Client(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=2, batch_size=16)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

fl.client.start_numpy_client(server_address="localhost:8080", client=Client())