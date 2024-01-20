from collections.abc import Callable

from magvlt.datamodules.datamodule import DataModule
from magvlt.datamodules.transforms import DalleTransform


def build_datamodule(cfg, **kwargs):
    return DataModule(cfg, **kwargs)


def build_transform(cfg, split: str) -> Callable:
    if cfg.dataset.transform.type == "dalle-vqvae":
        return DalleTransform(cfg=cfg, split=split)
    else:
        raise NotImplementedError(f"{cfg.dataset.transform.type} is not implemented..")

