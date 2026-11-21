from model_service.components import Batch, Step


class Second(Step):
    """Test step whose name another module already uses."""

    name = "twin"

    def __call__(self, batch: Batch) -> Batch:
        """Pass the batch through."""
        return batch
