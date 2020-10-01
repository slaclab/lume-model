"""
Adaptation of tensorflow tutorial: https://www.tensorflow.org/tutorials/estimator/premade
"""
from lume_model.utils import model_from_yaml

model = model_from_yaml("examples/files/iris_config.yaml")
model.random_evaluate()
