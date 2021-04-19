""" Early Stopping Strategies """
class EarlyStoppingStrategy(object):
    """
    Base class for early stopping strategies
    """

    def __init__(self):
        pass

    def evaluate(self, *args, **kwargs) -> bool:
        # must return True or False - when True, the executor should stop training the NN
        raise NotImplementedError


class TrainingLossEarlyStoppingStrategy(EarlyStoppingStrategy):

    def __init__(self):
        super().__init__()

    def evaluate(self, current_loss: float, previous_loss: float) -> bool:
        if previous_loss < current_loss:
            return True
        return False


