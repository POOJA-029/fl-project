import flwr as fl

strategy = fl.server.strategy.FedAvg()

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)