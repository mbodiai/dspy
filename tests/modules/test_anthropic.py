import pytest
from unittest.mock import patch, MagicMock
import dsp.modules.anthropic as dsp_anthropic
from dsp.modules.anthropic import Claude, backoff_hdlr, giveup_hdlr, RateLimitError
from dsp.primitives.vision import Image

mock_response = MagicMock()
mock_response.usage = MagicMock(input_tokens=10, output_tokens=10)
mock_response.content = [MagicMock(text="Test completion")]


class MockMessages:
    @staticmethod
    def create(**kwargs):
        return mock_response

class MockAnthropic:
    messages = MockMessages


# Mock Image object
mock_image = MagicMock(spec=Image)
mock_image.encoding = "png"
mock_image.base64 = "base64_image_data"

# Test the basic_request method
def test_basic_request():
    with patch('dsp_anthropic.Anthropic.messages.create') as mock_create:
        mock_create.return_value = mock_response
        claude = Claude(api_key="test_api_key")
        response = claude.basic_request(prompt="Test prompt")
        assert response == mock_response
        mock_create.assert_called_once()
        
# Test the basic_request method with image input
def test_basic_request_with_image():
    with patch('Claude.Anthropic.messages.create') as mock_create:
        mock_create.return_value = mock_response
        claude = Claude(api_key="test_api_key")
        response = claude.basic_request(prompt="Test prompt", image=mock_image)
        assert response == mock_response
        mock_create.assert_called_once()
        # Check if the image data was included in the request
        assert mock_create.call_args[1]['messages'][0]['content'][1]['source']['data'] == "base64_image_data"

# Test the __call__ method with image input
def test_call_method_with_image():
    with patch('Claude.request') as mock_request:
        mock_request.return_value = mock_response
        claude = Claude(api_key="test_api_key")
        completions = claude(prompt="Test prompt", image=mock_image)
        assert completions == ["Test completion"]
        mock_request.assert_called_once()
        # Check if the image data was included in the request
        assert mock_request.call_args[1]['image'] == mock_image

# Test the request method with backoff and error handling
def test_request_with_backoff():
    with patch('anthropic.Anthropic.messages.create') as mock_create:
        mock_create.side_effect = [RateLimitError("Rate limit exceeded"), mock_response]
        claude = Claude(api_key="test_api_key")
        response = claude.request(prompt="Test prompt")
        assert response == mock_response
        assert mock_create.call_count == 2

# Test the __call__ method
def test_call_method():
    with patch('dsp.modules.claude.Claude.request') as mock_request:
        mock_request.return_value = mock_response
        claude = Claude(api_key="test_api_key")
        completions = claude(prompt="Test prompt")
        assert completions == ["Test completion"]
        mock_request.assert_called_once()

# Test the logging functionality
def test_log_usage(caplog):
    claude = Claude(api_key="test_api_key")
    claude.log_usage(mock_response)
    assert "20" in caplog.text

# Test the backoff handler
def test_backoff_handler(caplog):
    details = {
        "wait": 1.0,
        "tries": 1,
        "target": "test_target",
        "kwargs": {"test": "value"}
    }
    backoff_hdlr(details)
    assert "Backing off 1.0 seconds after 1 tries calling function test_target with kwargs {'test': 'value'}" in caplog.text

# Test the giveup handler
def test_giveup_handler():
    details = MagicMock()
    details.message = "Rate limit exceeded"
    assert not giveup_hdlr(details)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, '-vv'])
