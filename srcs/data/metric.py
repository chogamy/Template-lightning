from torchmetrics import Metric


class data(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.preds = []
        self.targets = []

    def update(self):
        pass

    def compute(self):
        pass


METRIC = {"data": data}
