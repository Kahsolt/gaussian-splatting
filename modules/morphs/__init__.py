from pathlib import Path
MODEL_MORPHS = [fp.stem for fp in Path(__file__).parent.iterdir() if fp.is_dir() and not fp.name.startswith('_')]

from .gs.model import GaussianModel as GaussianModel_gs
from .mlp_gs.model import GaussianModel as GaussianModel_mlp_gs
from .cd_gs.model import GaussianModel as GaussianModel_cd_gs
from .if_gs.model import MutilFreqGaussianModel as GaussianModel_if_gs
from .dev.model import GaussianModel as GaussianModel_dev

from typing import Union
GaussianModel = Union[
  GaussianModel_gs,
  GaussianModel_mlp_gs,
  GaussianModel_cd_gs,
  GaussianModel_if_gs,
  GaussianModel_dev,
]
