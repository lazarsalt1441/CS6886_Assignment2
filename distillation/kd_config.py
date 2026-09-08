
# ============================================================
# Knowledge Distillation configuration
# ============================================================

# Student widths to train sequentially.
STUDENT_WIDTHS = [0.50, 0.35]

# Distillation hyperparameters
TEMPERATURE = 4.0

# Final loss:
#   alpha * hard-label CE
# + (1-alpha) * distillation KL
ALPHA = 0.30

# Training
NUM_EPOCHS = 100
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Cosine schedule over all epochs
USE_COSINE_SCHEDULER = True

# Optional warmup
WARMUP_EPOCHS = 5

# Gradient clipping; None disables it
MAX_GRAD_NORM = 5.0

# Checkpoint/logging
KD_OUTPUT_DIR = "kd_outputs"
KD_RESULTS_CSV = "kd_results.csv"

# Resume each student from last checkpoint if available
RESUME = True

# Save a latest checkpoint every epoch so overnight runs are safe
SAVE_EVERY_EPOCH = True

# If True, also train a hard-label-only student for comparison.
# This roughly doubles runtime, so leave False for the overnight run.
RUN_HARD_LABEL_BASELINE = False

# Reproducibility
KD_SEED = 42
