"""Training: datasets, photometric loss (L1 + SSIM), Adam, densification, the trainer."""

from .adam import Adam
from .dataset import Dataset, View, load_colmap_dataset
from .densify import Densifier
from .loss import PhotometricLoss
from .trainer import TrainConfig, Trainer
