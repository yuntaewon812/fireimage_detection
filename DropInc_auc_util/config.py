from dataclasses import dataclass


@dataclass(frozen=True)
class DropIncConfig:
    seed: int = 1004
    class_names: tuple = ('fireimage',)
    base_model_path: str = "./model_save"
    out_dir: str = "./results_DROP_INC"

    img_size: int = 224
    folds: tuple = (0, 1, 2)
    n_splits: int = 3
