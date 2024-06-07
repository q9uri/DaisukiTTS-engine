# flake8: noqa

import warnings

from pydantic.warnings import PydanticDeprecatedSince20

__version__ = "latest"

# PydanticDeprecatedSince20 警告を抑制
warnings.filterwarnings(
    action="ignore",
    category=PydanticDeprecatedSince20,
)

# PyTorch による UserWarning 警告を抑制
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)
