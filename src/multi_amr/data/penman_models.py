from penman.model import Model as PenmanModel
from penman.types import BasicTriple


class NoInvertModel(PenmanModel):
    def invert(self, triple: BasicTriple) -> BasicTriple:
        """Override default behavior and do not invert.
        """
        return triple


no_invert_model = NoInvertModel()
