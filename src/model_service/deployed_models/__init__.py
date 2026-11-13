"""Models as the service sees them.

- catalog.py: the deployed models, read from the model YAML files.
- model_clients/: the clients, one module per contract, that call a model and check its answers.

What a model of each contract declares is contracts/model_declarations.py.
"""
