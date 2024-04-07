from pydantic import BaseModel, ConfigDict
from typing import Optional, Union, List, Any
import numpy as np
from mbodied.common.senses import Image

class BracketNotationType:
    def __init__(self, initial_data=None):
        """Initialize the object with a dictionary."""
        self.data = initial_data or {}

    def __getitem__(self, key):
        """Retrieve an item using bracket notation."""
        return self.data[key]

    def __setitem__(self, key, value):
        """Set an item using bracket notation."""
        self.data[key] = value

    def __iter__(self):
        """Return an iterator over the keys of the dictionary."""
        return iter(self.data)

    def __contains__(self, key):
        """Check if a key is in the dictionary."""
        return key in self.data

    def __len__(self):
        """Return the number of items in the dictionary."""
        return len(self.data)

    def __repr__(self):
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}({self.data})"

class ArrayType(BracketNotationType):
class TypedArray(BracketNotationType):
    def __init__(self, initial_data=None, dtype=None):
        """Initialize the object with a dictionary and a data type."""
        super().__init__(initial_data)
        self.dtype = dtype

    def __setitem__(self, key, value):
        """Set an item using bracket notation and cast it to the data type."""
        self.data[key] = self.dtype(value)

    def __repr__(self):
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}({self.data}, dtype={self.dtype})"


class Modality:
  # can be text, image, audio, video, etc.
  pass
  

class MMPrompt(BaseModel):
    model_config: ConfigDict
    text: str
    texts = Optional[List[str]] = None
    image: Optional[Image] = None
    images: Optional[List[Image]] = None
    multimodal: Optional[List[Union[str, Image]]] = None
    
    def __init__(self, arg: Any=None, **kwargs: Any):
        super().__init__(**kwargs)
        
        if isinstance(arg, str):
            self.text = arg
        elif Image.su
    
    def ask(question, backends: List[BackendAgent], strategy = 'waterfall'): # waterfall, divide_and_conquer):
     if question.is_answered():
       self.pass_backward(question.response)
     else:
       self.pass_forward(question, strategy, backends)
         
          
          
          
    
    def pass_forward(self,ask, strategy):
      if strategy == 'waterfall':
        found_answer = self.supervise(ask.response)
        for i, b in enumerate(backends):
          response = self.backend.ask(self.supervise)
          if response.is_correct():
            self.pass_backward(response)
            break
    
    def pass_backward(self,ask):
      pass
    
    def is_answered(self,ask):
      pass
    
    def is_correcdt(self,ask):
      pass
      
class AskAnswer(BaseModel):
    ask: MMPrompt
    

