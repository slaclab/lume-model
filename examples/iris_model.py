"""
Adaptation of tensorflow tutorial: https://www.tensorflow.org/tutorials/estimator/premade
"""
from lume_model.utils import model_from_yaml

with open("examples/files/iris_config.yaml", "r") as f:
    model = model_from_yaml(f)

model.random_evaluate()
