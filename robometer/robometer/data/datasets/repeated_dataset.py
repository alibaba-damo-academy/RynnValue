"""Minimal repeat wrapper for training; exposes .dataset for checkpointing."""

from torch.utils.data import Dataset


class RepeatedDataset(Dataset):
    """Wraps a dataset and repeats it for a fixed number of effective steps (or a large default)."""

    def __init__(self, dataset: Dataset, num_repeats: int = 1000):
        self.dataset = dataset
        self._len = len(dataset)
        self.num_repeats = num_repeats
        if self._len == 0:
            self._effective_len = 0
        else:
            self._effective_len = min(self._len * num_repeats, self._len * 10_000)

    def __len__(self) -> int:
        return self._effective_len if self._effective_len > 0 else 1

    def __getitem__(self, index: int):
        if self._len == 0:
            raise IndexError("Underlying dataset is empty")
        return self.dataset[index % self._len]
