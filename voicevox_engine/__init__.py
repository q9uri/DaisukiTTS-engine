# flake8: noqa

import warnings

from pydantic.warnings import PydanticDeprecatedSince20

__version__ = "latest"

# PydanticDeprecatedSince20 警告を抑制
warnings.filterwarnings(
    action="ignore",
    category=PydanticDeprecatedSince20,
)

# pyannote.audio による UserWarning 警告を抑制
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.",
)
