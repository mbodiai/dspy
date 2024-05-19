import logging
import os
from typing import Any, Optional

import backoff
from mbodied.types.sense.vision import Image, SupportsImage

from dsp.modules.vlm import VLM

try:
    from anthropic import RateLimitError
except ImportError:
    RateLimitError = Exception

logger = logging.getLogger(__name__)
logger1 = logging.getLogger("anthropic.log")
logger2 = logging.getLogger("anthropic_light.log")

# Set levels
logger1.setLevel(logging.INFO)
logger2.setLevel(logging.INFO)

# Create file handlers
handler1 = logging.FileHandler('anthropic.log')
handler2 = logging.FileHandler('anthropic_light.log')

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the handlers
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)

# Add the handlers to the loggers
logger1.addHandler(handler1)
logger2.addHandler(handler2)
BASE_URL = "https://api.anthropic.com/v1/messages"


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/."""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


def giveup_hdlr(details):
    """Wrapper function that decides when to give up on retry."""
    if "rate limits" in details.message:
        return False
    return True

client = None  
class Claude(VLM):
    """Wrapper around anthropic's API. Supports both the Anthropic and Azure APIs."""

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)
        try:
            from anthropic import Anthropic
        except ImportError as err:
            raise ImportError(
                "Claude requires `pip install anthropic`.",
            ) from err

        self.provider = "anthropic"
        self.api_key = api_key = (
            os.environ.get("ANTHROPIC_API_KEY") if api_key is None else api_key
        )
        self.api_base = BASE_URL if api_base is None else api_base
        self.kwargs = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": min(kwargs.get("max_tokens", 4096), 4096),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", 1),
            "n": kwargs.pop("n", kwargs.pop("num_generations", 1)),
            **kwargs,
        }
        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []
        global client
        client = Anthropic(api_key=api_key)
        self.client = client

    def log_usage(self, response):
        """Log the total tokens from the Anthropic API response."""
        usage_data = response.usage
        if usage_data:
            total_tokens = usage_data.input_tokens + usage_data.output_tokens
            logger.info(f"{total_tokens}")

    def basic_request(self, prompt: str, image: SupportsImage = None, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        messages = kwargs.get("messages", [])
        if kwargs.get("n_past",False):
            history = self.history[-kwargs["n_past"]:]
            kwargs.pop("n_past")
            for h in history:
                messages += [{"role": "user", "content": h['prompt']},
                                {"role": "assistant", "content": h["response"].content}]
        content = []
        if kwargs.get("stored_image", None):
            content.append({"type": "text", "text": "Best recent view of the scene."})
            image = kwargs["stored_image"]
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image.encoding}",
                        "data": image.base64,
                    },
                },
            )
            kwargs.pop("stored_image")
        content += [{"type": "text", "text": prompt}]
        if image:
            image = Image(image) if not isinstance(image, Image) else image
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image.encoding}",
                        "data": image.base64,
                    },
                },
            )
        kwargs["messages"] = messages + [{"role": "user", "content": content}]
        logger1.info(f"sending kwargs: {kwargs}")
        kwargs.pop("n")
        kwargs.pop("stored_image", None)
        response = claude_request(**kwargs)
        history = {
            "prompt": prompt,
            # "image": image,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)
        light_kwargs = {k: v for k, v in kwargs.items() if "image" not in k}
        for k in light_kwargs:
            logger2.info(f"{k}: {light_kwargs[k]}")
            logger2.info(f"response: {response}")
        return response.content[0].text

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError,),
        max_time=1000,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Anthropic whilst handling API errors."""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self, prompt, only_completed=True, return_sorted=False, **kwargs,
    ):
        """Retrieves completions from Anthropic.

        Args:
            prompt (str): prompt to send to Anthropic
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[str]: list of completion choices
        """
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        # per eg here: https://docs.anthropic.com/claude/reference/messages-examples
        # max tokens can be used as a proxy to return smaller responses
        # so this cannot be a proper indicator for incomplete response unless it isnt the user-intent.
        n = kwargs.pop("n", 1)
        completions = []
        for _ in range(n):
            response = self.request(prompt, **kwargs)
            # TODO: Log llm usage instead of hardcoded openai usage
            # if dsp.settings.log_openai_usage:
            #     self.log_usage(response)
            if only_completed and response.stop_reason == "max_tokens":
                continue
            completions = [c.text for c in response.content]
        return completions


# @selective_cache(ignore_fields=["source"])
def cached_claude_request(**kwargs) -> Any:
    # print("kwargs:", kwargs)
    return client.messages.create(**kwargs)

# @functools.lru_cache(maxsize=None if cache_turn_on else 0)

# def cached_claude_request_wrapped(**kwargs) -> Any:
#     return cached_claude_request(**kwargs)


def claude_request(**kwargs) -> dict[str, Any]:
    return cached_claude_request(**kwargs)


# # @CacheMemory.cache
# def cached_gpt4vision_chat_request(**kwargs) -> Any:
#   if "stringify_request" in kwargs:
#     kwargs = json.loads(kwargs["stringify_request"])
#   return openai.chat.completions.create(**kwargs)


# # @functools.lru_cache(maxsize=None if cache_turn_on else 0)
# # @NotebookCacheMemory.cache
# def cached_gpt4vision_chat_request_wrapped(**kwargs) -> Any:
#   return cached_gpt4vision_chat_request(**kwargs)

# def chat_request(**kwargs) -> dict[str, Any]:
#   return cached_gpt4vision_chat_request_wrapped(**kwargs)
