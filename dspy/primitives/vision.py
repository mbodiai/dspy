import base64 as base64lib
import io
from typing import Optional

import numpy as np
import PIL.Image as PILImage
from pydantic import Field, model_validator, BaseModel, field_validator 
from typing import Optional, Tuple, Union, Any, Type

SupportsImage = Union[str, Image, np.ndarray, PILImage.Image]


class Image(BaseModel):
    """A class to represent an image. The image can be initialized with pixels, a base64 string, or a file path."""
    pixels: Optional[np.ndarray] = Field(None, exclude=True)
    base64: str = ''
    encoding: str = 'png'
    path: Optional[str] = None
    pil: Optional[PILImage] = Field(None, exclude=True)
    url: Optional[str] = None
    size: Optional[tuple[int, int]] = Field(None, exclude=True)

    
    @model_validator(mode='before')
    @classmethod
    def validate_image(cls, ):
        encoding = values.get('encoding', 'png')  # Get the encoding from the values
        values['encoding'] = encoding
        if 'pixels' in values and values['pixels'] is not None:
            pixels = values['pixels']
            values['size'] = pixels.shape[:2]
            values['base64'] = cls.encode_pixels(pixels, encoding)
        elif 'path' in values and values['path'] is not None:
            with PILImage.open(values['path']) as img:
                pixels = np.array(img)
                values['pixels'] = pixels
                values['size'] = pixels.shape[:2]
                values['base64'] = cls.encode_pixels(pixels, encoding)
        elif 'base64' in values and values['base64'] is not None:
            pixels = cls.decode_base64(values['base64'], encoding)
            values['pixels'] = pixels
            values['size'] = pixels.shape[:2]

        return values

    @classmethod
    def encode_pixels(cls, pixels, encoding):
        # Convert the pixels array to an image and encode it in base64
        img = PILImage.fromarray(pixels)
        buffered = io.BytesIO()
        if encoding.lower() == 'jpg':
            encoding = 'jpeg'
        if encoding.lower() not in ['png', 'jpeg']:
            raise ValueError(f"Unsupported encoding '{encoding}'. Supported encodings are 'png' and 'jpeg'.")
        img.save(buffered, format=encoding.upper())
        return base64lib.b64encode(buffered.getvalue()).decode()

    @classmethod
    def decode_base64(cls, base64_str, encoding):
        # Decode the base64 string to an image
        img_data = base64lib.b64decode(base64_str)
        img = PILImage.open(io.BytesIO(img_data))
        if img.format.lower() != encoding.lower():
            raise ValueError(f"Expected image encoding '{encoding}', got '{img.format}'")
        return np.array(img)


    @property
    def image_array(self) -> Optional[np.ndarray]:
        if self.pixels is not None:
            return self.pixels
        if self.base64:
            return self.decode_base64(self.base64, self.encoding)
        return None

    @property
    def image_base64(self) -> str:
        if self.base64:
            return self.base64
        if self.pixels is not None:
            return self.encode_pixels(self.pixels, self.encoding)
        return ''

    def base64_encode(self, encoding: str = 'png') -> str:
        return self.encode_pixels(self.pixels, encoding)

    def save(self, path: str):
        img = PILImage.fromarray(self.pixels)
        img.save(path)
        
    def model_dump(self, *args, **kwargs):
        return {
            'base64': self.image_base64,
        }