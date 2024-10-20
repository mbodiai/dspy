import dsp
from dspy.dsp.modules.hf_client import ChatModuleClient, HFClientSGLang, HFClientVLLM, HFServerTGI

from dspy.predict import *
from dspy.primitives import *
from dspy.retrieve import *
from dspy.signatures import *
from dspy.utils.logging import logger, set_log_output

# Functional must be imported after primitives, predict and signatures
from dspy.functional import *  # isort: skip
from dspy.evaluate import Evaluate # isort: skip
from dspy.clients import * # isort: skip
from dspy.adapters import * # isort: skip
from dsp import ColBERTv2RetrieverLocal # isort: skip
settings = dsp.settings

# LM = dsp.LM

AzureOpenAI = dsp.AzureOpenAI
OpenAI = dsp.GPT3
MultiOpenAI = dsp.MultiOpenAI
Mistral = dsp.Mistral
Databricks = dsp.Databricks
Cohere = dsp.Cohere
ColBERTv2 = dsp.ColBERTv2

Pyserini = dsp.PyseriniRetriever
Clarifai = dsp.ClarifaiLLM
CloudflareAI = dsp.CloudflareAI
Google = dsp.Google
GoogleVertexAI = dsp.GoogleVertexAI
GROQ = dsp.GroqLM
Snowflake = dsp.Snowflake
Claude = dsp.Claude

HFClientTGI = dsp.HFClientTGI

Anyscale = dsp.Anyscale
Together = dsp.Together
HFModel = dsp.HFModel
OllamaLocal = dsp.OllamaLocal
LlamaCpp = dsp.LlamaCpp

Bedrock = dsp.Bedrock
Sagemaker = dsp.Sagemaker
AWSModel = dsp.AWSModel
AWSMistral = dsp.AWSMistral
AWSAnthropic = dsp.AWSAnthropic
AWSMeta = dsp.AWSMeta

Watsonx = dsp.Watsonx
PremAI = dsp.PremAI

You = dsp.You

configure = settings.configure
context = settings.context


import dspy.teleprompt

LabeledFewShot = dspy.teleprompt.LabeledFewShot
BootstrapFewShot = dspy.teleprompt.BootstrapFewShot
BootstrapFewShotWithRandomSearch = dspy.teleprompt.BootstrapFewShotWithRandomSearch
BootstrapRS = dspy.teleprompt.BootstrapFewShotWithRandomSearch
COPRO = dspy.teleprompt.COPRO
MIPROv2 = dspy.teleprompt.MIPROv2
Ensemble = dspy.teleprompt.Ensemble


# TODO: Consider if this should access settings.lm *or* a list that's shared across all LMs in the program.
def inspect_history(*args, **kwargs):
    return settings.lm.inspect_history(*args, **kwargs)