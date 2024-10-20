"""
## DSPY TODO:

- Module Graph
- Avatar optimizer
- Synthetic Data Gen
-
"""

# Example usage of the ModuleGraph class:
import os

from dotenv import load_dotenv

import dspy
from dspygraph import ModuleGraph
from colbert.infra.config import ColBERTConfig
load_dotenv()

# Configuration of dspy models
llm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"], max_tokens=100)
from rich.pretty import pprint

pprint(ColBERTConfig.__dict__)
colbertv2_wiki = dspy.ColBERTv2RetrieverLocal(
    url="http://localhost:5000",
    passages=["The capital of France is Paris.", "The capital of Germany is Berlin."]
)

dspy.settings.configure(lm=llm, rm=colbertv2_wiki)


class GenerateAnswer(dspy.Signature):
    """Answer with long and detailed answers."""

    context = dspy.InputField(desc="may content relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 10 and 50 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


rag_system = RAG()
graph = ModuleGraph("RAG", rag_system)

graph.render_graph()
