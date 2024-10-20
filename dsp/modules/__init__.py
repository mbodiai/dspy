# from dspy.dsp.anthropic import Claude

if not globals().get('DSP_MODULES_IMPORTED'):
  from dspy.dsp.modules.dspyanthropic import Claude
  globals()['DSP_MODULES_IMPORTED'] = True
from dspy.dsp.modules.aws_models import AWSAnthropic, AWSMeta, AWSMistral, AWSModel

# Below is obsolete. It has been replaced with Bedrock class in dsp/modules/aws_providers.py
# from dspy.dsp.bedrock import *
from dspy.dsp.modules.aws_providers import Bedrock, Sagemaker
from dspy.dsp.modules.azure_openai import AzureOpenAI
from dspy.dsp.modules.cache_utils import *
from dspy.dsp.modules.clarifai import *
from dspy.dsp.modules.cloudflare import *
from dspy.dsp.modules.cohere import *
from dspy.dsp.modules.databricks import *
from dspy.dsp.modules.dummy_lm import *
from dspy.dsp.modules.google import *
from dspy.dsp.modules.google_vertex_ai import *
from dspy.dsp.modules.gpt3 import *
from dspy.dsp.modules.groq_client import *
from dspy.dsp.modules.hf import HFModel
from dspy.dsp.modules.hf_client import Anyscale, HFClientTGI, Together
from dspy.dsp.modules.llama import *
from dspy.dsp.modules.mistral import *
from dspy.dsp.modules.ollama import *
from dspy.dsp.modules.multi_openai import MultiOpenAI
from dspy.dsp.modules.premai import PremAI
from dspy.dsp.modules.pyserini import *
from dspy.dsp.modules.sbert import *
from dspy.dsp.modules.sentence_vectorizer import *
from dspy.dsp.modules.snowflake import *
from dspy.dsp.modules.tensorrt_llm import TensorRTModel
from dspy.dsp.modules.watsonx import *
from dspy.dsp.modules.you import You
