from abc import ABC, abstractmethod

class TrainAlgo(ABC):
    @abstractmethod
    def train(self, model, dataloader, params):
        pass