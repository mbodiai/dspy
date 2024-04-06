from pydantic import BaseModel
from typing import Optional, Union, List, Any
from PIL import Image as PILImage
import numpy as np


MultiModalInput = Union[str, PILImage.Image, np.ndarray, List[Union[str, PILImage.Image, np.ndarray]]]