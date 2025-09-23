from rfdetr.detr import RFDETRBase
from pathlib import Path

model = RFDETRBase()
cwd = Path(__file__).parent

model.train(
    batch_size = 4,
    dataset_dir = cwd / "dataset",
    early_stopping = True,
    epochs = 10,
    grad_accum_steps = 4,
    lr = 1e-4,
    output_dir = cwd / "output",
    run = "RUN_1_NAME",
    wandb = True,
)
