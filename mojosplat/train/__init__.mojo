"""Training: datasets, photometric loss (L1 + SSIM), Adam, densification, the trainer."""

from .adam import Adam
from .dataset import Dataset, View, load_colmap_dataset
from .densify import ClassicDensification, Densifier, DensifyConfig
from .loss import PhotometricLoss
from .strategy import Densification
from .trainer import TrainConfig, Trainer
