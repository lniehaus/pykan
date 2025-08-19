import torch
import numpy as np

from enum import Enum
from sklearn.decomposition import PCA

from dltb.graph.graph import NumpyProbe


class ProbeMode(Enum):
    TRAIN = "train"
    PREDICT = "predict"


class LinearClassifierProbe(NumpyProbe):
    """Linear classifier probe which can train a linear classifier on layer input or output."""
    def __init__(
        self,
        classifier,
        classes: np.ndarray,
        feature_processor=None,
        attach_to_input: bool = False,
        batch_size: int = 32
    ):
        self.classifier = classifier
        self.classes = classes

        self.attach_to_input = attach_to_input

        self._mode: ProbeMode = ProbeMode.TRAIN  # train || predict
        self.labels: torch.Tensor
        self.evaluations = None
        self.feature_processor = feature_processor
        self.name = ""
        self.batch_size = batch_size

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        if ProbeMode(new_mode) not in ProbeMode:
            raise ValueError("Unsupported mode!")
        self._mode = new_mode

    def set_mode(self, mode: str) -> None:
        self.mode = ProbeMode(mode)

    @property
    def predictions(self):
        return self._predictions

    @property
    def accuracy(self) -> float:
        return np.mean(self.evaluations)

    def measure(self, node, batched_input: np.ndarray, batched_output: np.ndarray):
        if self.attach_to_input:
            feature_input = batched_input
        else:
            feature_input = batched_output

        # FIXME: Currently mitigating shape missmatch in last batch by dropping it
        if feature_input.shape[0] < self.batch_size:
            return
        self.name = node.name
        features = self.process_features(feature_input)
        if self.mode == ProbeMode.TRAIN:
            self._fit(features)
        elif self.mode == ProbeMode.PREDICT:
            self._predict(features)

    def _fit(self, features: np.ndarray) -> None:
        self.classifier.partial_fit(features, self.labels, classes=np.identity(10))

    def _predict(self, features: np.ndarray) -> None:
        if self.evaluations is None:
            self.evaluations = self.classifier.score(features, self.labels)
        else:
            self.evaluations = np.append(
                self.evaluations, self.classifier.score(features, self.labels)
            )

    def process_features(self, input_features: np.ndarray) -> np.ndarray:
        """Flatten input features if their dimensionality is too high"""
        if self.feature_processor is not None:
            return self.feature_processor(input_features)
        elif input_features.ndim > 2:
            return input_features.flatten().reshape(input_features.shape[0], -1)
        return input_features


def pca_processor(input_features: np.ndarray) -> np.ndarray:
    # FIXME: Do not calculate the PCA over each batch but incrementally
    if input_features.ndim > 2:
        features = input_features.flatten().reshape(input_features.shape[0], -1)
    else:
        features = input_features    

    pca = PCA(32)
    reduction = pca.fit_transform(features)
    print(pca.explained_variance_ratio_)
    return reduction.flatten().reshape(input_features.shape[0], -1)



if __name__ == "__main__":
    from tqdm import tqdm
    from sklearn.linear_model import SGDClassifier

    from helpers import get_lenet, get_cifar10

    graph = get_lenet("data/cifar_net.pth")
    trainloader, testloader = get_cifar10()

    probe = LinearClassifierProbe(
    classifier=SGDClassifier(
        loss="log_loss", learning_rate="constant", eta0=0.01),
        classes = np.identity(10),
        # feature_processor=pca_processor
    )

    graph.nodes[5].add_probe(probe)


    for data in tqdm(trainloader, "Training"):
        input_images, input_labels = data

        probe.labels = input_labels.numpy()
        graph.forward(input_images)