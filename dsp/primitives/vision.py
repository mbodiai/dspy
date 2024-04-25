import base64 as base64lib
import importlib
import io
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
from PIL import Image as PILModule
from PIL.Image import Image as PILImage
from pydantic import AnyUrl, Base64Str, BaseModel, ConfigDict, Field, FilePath, InstanceOf, model_validator
from typing_extensions import Literal

SupportsImage =  Union[np.ndarray, PILImage, Base64Str, AnyUrl, FilePath]


class Image(BaseModel):
  """A class to represent an image. The image can be initialized with a numpy array, a base64 string, or a file path.

  Attributes:
      array (Optional[np.ndarray]): The image represented as a NumPy array.
      base64 (str): The image encoded as a base64 string.
      encoding (str): The format used for encoding the image when converting to base64.
      path (Optional[str]): The file path to the image if initialized from a file.
      pil (Optional[PILImage]): The image represented as a PIL Image object.
      url (Optional[str]): The URL to the image if initialized from a URL.
      size (Optional[tuple[int, int]]): The size of the image as a (width, height) tuple.

  Example:
      >>> from vision import Image
      >>> import numpy as np
      >>> # Initialize with a NumPy array
      >>> arr = np.zeros((100, 100, 3), dtype=np.uint8)
      >>> img_from_array = Image(arr)
      >>> # Initialize with a base64 string
      >>> base64_str = 'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='
      >>> img_from_base64 = Image(base64_str)
      >>> # Initialize with a file path
      >>> img_from_path = Image('path/to/image.png')
      >>> # Access the PIL Image object
      >>> pil_image = img_from_array.pil
  """
  model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

  array: Optional[np.ndarray] = Field(None, exclude=True)
  base64: Optional[InstanceOf[Base64Str]] = Field(None, exclude=True)
  path: Optional[FilePath] = Field(None, exclude=True)
  pil: Optional[InstanceOf[PILImage]] = Field(None, exclude=True)
  url: Optional[Union[InstanceOf[AnyUrl], str]] = Field(None, exclude=True)
  size: Optional[tuple[int, int]] = Field(None, exclude=True)
  encoding: Literal['png', 'jpeg', 'jpg', 'bmp', 'gif'] = Field('jpeg', exclude=False)
  
  @classmethod
  def supports(cls, arg: SupportsImage) -> bool:
    return isinstance(arg, (np.ndarray, PILImage)) or Path(arg).exists() or arg.startswith('data:image') or urlparse(arg).scheme

  def __init__(self, arg:SupportsImage=None, **kwargs):
    if arg is not None:
      if isinstance(arg, str):
        if isinstance(arg, AnyUrl):
          kwargs['url'] = arg
        elif Path(arg).exists():
          kwargs['path'] = arg
        else:
          kwargs['base64'] = arg
      elif isinstance(arg, Path):
        kwargs['path'] = str(arg)
      elif isinstance(arg, np.ndarray):
        kwargs['array'] = arg
      elif isinstance(arg, PILImage):
        kwargs['pil'] = arg
      elif isinstance(arg, Image):
        # Overwrite an Image instance with the new kwargs
        kwargs.update({'array': arg.array})
      else:
        raise ValueError(f"Unsupported argument type '{type(arg)}'.")
    super().__init__(**kwargs)

  def __repr__(self):
    """Return a string representation of the image."""
    return f"Image(base64={self.base64[:10]}..., encoding={self.encoding}, size={self.size})"

  def __str__(self):
    """Return a string representation of the image."""
    return self.__repr__()

  @staticmethod
  def from_base64(base64_str: str, encoding: str) -> dict:
    """Decodes a base64 string to create an Image instance.

    Args:
        base64_str (str): The base64 string to decode.
        encoding (str): The format used for encoding the image when converting to base64.

    Returns:
        Image: An instance of the Image class with populated fields.
    """
    image_data = base64lib.b64decode(base64_str)
    image = PILModule.open(io.BytesIO(image_data)).convert('RGB')
    return Image.from_pil(image, encoding)

  @staticmethod
  def from_pil(image: PILImage, encoding: str) -> dict:
    """Creates an Image instance from a PIL image.

    Args:
        image (PIL.Image.Image): The source PIL image from which to create the Image instance.
        encoding (str): The format used for encoding the image when converting to base64.

    Returns:
        Image: An instance of the Image class with populated fields.
      """
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format=encoding.upper())
    base64_encoded = base64lib.b64encode(buffer.getvalue()).decode('utf-8')
    data_url = f"data:image/{encoding};base64,{base64_encoded}"

    return {
        'array': np.array(image),
        'base64': base64_encoded,
        'pil': image,
        'size': image.size,
        'url': data_url,
        'encoding': encoding.lower(),
    }

  @staticmethod
  def load_image(url: str) -> PILImage:
    """Downloads an image from a URL or decodes it from a base64 data URI.

    Args:
        url (str): The URL of the image to download, or a base64 data URI.

    Returns:
        PIL.Image.Image: The downloaded and decoded image as a PIL Image object.
      """
    if url.startswith('data:image'):
      # Extract the base64 part of the data URI
      base64_str = url.split(';base64', 1)[1]
      image_data = base64lib.b64decode(base64_str)
    else:
      # Open the URL and read the image data
      with urlopen(url) as response:
        image_data = response.read()

    # Convert the image data to a PIL Image
    return PILModule.open(io.BytesIO(image_data)).convert('RGB')

  @model_validator(mode='before')
  @classmethod
  def validate_kwargs(cls, values) -> dict:
    """Validates and transforms input data before model initialization.

    Ensures that all values are not None and are consistent.

    Args:
        values (dict): The input data to validate.

    Returns:
        dict: The validated and possibly transformed input data.
      """
    provided_fields = {
        k: v
        for k, v in values.items()
        if v is not None and k in ['array', 'base64', 'path', 'pil', 'url']
    }
    if len(provided_fields) > 1:
      raise ValueError("Multiple image sources provided; only one is allowed.")

      # Initialize all fields to None or their default values
    validated_values = {
        'array': None,
        'base64': '',
        'encoding': values.get('encoding', 'jpeg'),
        'path': None,
        'pil': None,
        'url': None,
        'size': None,
    }

    if 'path' in values:
      # Load the image and convert if the file extension does not match the desired encoding
      image = PILModule.open(values['path']).convert('RGB')
      validated_values['path'] = values['path']
      validated_values.update(cls.from_pil(image, validated_values['encoding']))

    # Convert to PIL image and populate other fields
    elif 'array' in values:
      image = PILModule.fromarray(values['array']).convert('RGB')
      validated_values.update(cls.from_pil(image, validated_values['encoding']))

    elif 'pil' in values:
      validated_values.update(cls.from_pil(values['pil'], validated_values['encoding']))

    # If 'base64' is provided, decode and populate other fields
    elif 'base64' in values:
        validated_values.update(cls.from_base64(values['base64'], validated_values['encoding']))

    # If 'url' is provided, download the image and populate other fields
    elif 'url' in values:
      image = cls.load_image(values['url'])

      # Determine the encoding based on the URL's file extension
      url_path = urlparse(values['url']).path
      file_extension = Path(url_path).suffix[1:].lower()
      validated_values['encoding'] = file_extension
      validated_values.update(cls.from_pil(image, validated_values['encoding']))
      validated_values['url'] = values['url']

    if validated_values['encoding'] not in ['png', 'jpeg', 'jpg', 'bmp', 'gif']:
      raise ValueError("The 'encoding' must be a valid image format (png, jpeg, jpg, bmp, gif).")

    return validated_values

  def save(self, path: str) -> None:
    self.pil.save(path)

  def show(self) -> None:
    importlib.import_module('matplotlib.pyplot').imshow(self.pil)


