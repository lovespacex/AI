"""A small Chinese chat language model trained from random initialization."""

from .config import ModelConfig
from .model import MiniChatModel

__all__ = ["MiniChatModel", "ModelConfig"]

