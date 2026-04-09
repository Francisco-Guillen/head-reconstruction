from abc import ABC, abstractmethod


class Stage(ABC):
    """Base class for all pipeline stages."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, context: dict) -> dict:
        from utils.timing import timer
        with timer(self.name):
            output = self.run(context)
            if not isinstance(output, dict):
                raise TypeError(f"{self.name} must return a dict")
            return output

    @abstractmethod
    def run(self, context: dict) -> dict:
        pass

    def __repr__(self):
        return f"<Stage: {self.name}>"
