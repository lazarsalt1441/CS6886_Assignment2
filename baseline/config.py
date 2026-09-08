import torch

# -------------------------
# Reproducibility
# -------------------------
SEED = 42

# -------------------------
# Dataset
# -------------------------
DATA_DIR = "./"
IMAGE_SIZE = 96
NUM_CLASSES = 10

# -------------------------
# Training
# -------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 30

LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_WORKERS = 2

# -------------------------
# Model saving
# -------------------------
MODEL_PATH = "best_mobilenetv2_cifar10.pth"

# -------------------------
# Device
# -------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)