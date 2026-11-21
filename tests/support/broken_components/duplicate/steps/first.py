from model_service.components import Batch, Step


class First(Step):
    """Test step registered under a name another module also uses."""

    name = "twin"

    def __call__(self, batch: Batch) -> Batch:
        """Pass the batch through."""
        return batch
