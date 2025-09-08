import lightning as L
import nnue_dataset
import features
from feature_set import FeatureSet
from torch.utils.data import DataLoader, Dataset


class NNUEDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_filename: str,
        val_filename: str,
        features_name: str,
        num_workers: int,
        batch_size: int,
        filtered: bool,
        random_fen_skipping: bool,
        epoch_size: int,
    ):
        super().__init__()
        self.train_filename = train_filename
        self.val_filename = val_filename
        self.feature_set = features.get_feature_set_from_name(features_name)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.epoch_size = epoch_size
        self.val_size = 1000000

    def setup(self, stage: str):
        self.train_infinite = nnue_dataset.SparseBatchDataset(
            self.feature_set,
            self.train_filename,
            self.batch_size,
            num_workers=self.num_workers,
            filtered=self.filtered,
            random_fen_skipping=self.random_fen_skipping,
        )
        self.val_infinite = nnue_dataset.SparseBatchDataset(
            self.feature_set,
            self.val_filename,
            self.batch_size,
            filtered=self.filtered,
            random_fen_skipping=self.random_fen_skipping,
        )

    def train_dataloader(self):
        # num_workers has to be 0 for sparse, and 1 for dense
        # it currently cannot work in parallel mode but it shouldn't need to
        return DataLoader(
            nnue_dataset.FixedNumBatchesDataset(
                self.train_infinite,
                (self.epoch_size + self.batch_size - 1) // self.batch_size,
            ),
            batch_size=None,
            batch_sampler=None,
        )

    def val_dataloader(self):
        # num_workers has to be 0 for sparse, and 1 for dense
        # it currently cannot work in parallel mode but it shouldn't need to
        return DataLoader(
            nnue_dataset.FixedNumBatchesDataset(
                self.val_infinite,
                (self.val_size + self.batch_size - 1) // self.batch_size,
            ),
            batch_size=None,
            batch_sampler=None,
        )
