# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path
from IPython import get_ipython

# %% [markdown]
# <img src="../../docs/images/DSPy8.png" alt="DSPy7 Image" height="150"/>
# 
# 
# ## **DSPy Assertions**: Asserting Computational Constraints on Foundation 
# 
# ### **LongFormQA**: Generating long-form length responses to answer questions
# %% [markdown]
# [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/longformqa/longformqa_assertions.ipynb)
# 
# This notebook builds upon the foundational concepts of the **DSPy** framework, as introduced in our previous tutorial (see [intro.ipynb](./intro.ipynb) for a refresher). DSPy overs a novel programming-centric approach to utilizing language and retrieval models. It offers a unique blend of prompting, reasoning, fine-tuning, and tool augmentation, all encapsulated under a minimalistic Python syntax. 
# 
# In this advancement of DSPy, we introduce **Assertions**, a feature with the capability to declare computational constraints within DSPy programs. This allows programmers to specify natural-language rules for valid outputs, guiding the behavior of language model calls during both compiling and inference stages. 
# 
# Our approach harnesses Pythonic style of assertions while meshing backtracking logic to ensure autonomous self-correction and refinement of language model calls. By accounting for past outputs and passing forward relevant feedback and guidelines for self-correction, this feature offers a significant leap in DSPy with enhanced control over program behavior.
# 
# This notebook demonstrates the utility of assertions on specific downstream examples, extending the Multi-Hop Question-Answering task from the [intro.ipynb](./intro.ipynb) to long-form paragraph generation with citations to answer questions. We demonstrate the performance benefits of integrating assertions to ensure the inclusion of citations in a predefined format and the faithfulness of generated text to its cited references. 
# %% [markdown]
# ### 0] Setting Up
# Let's begin by setting things up.
# %% [markdown]
# We will install **DSPy** if it's not there already.

# %%


import sys
import os
import regex as re


if repo_path not in sys.path:
    sys.path.append(repo_path)


import pkg_resources # Install the package if it's not installed
if "dspy-ai" not in {pkg.key for pkg in pkg_resources.working_set}:
    get_ipython().system('pip install -U pip')
    get_ipython().system('pip install dspy-ai')
    get_ipython().system('pip install openai~=0.28.1')
    get_ipython().system('pip install -e $repo_path')

import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA

from dspy.optimizers import BootstrapFewShotWithRandomSearch
from dsp.utils import normalize_text
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

cwd = Path.cwd() / Path('dspy/examples/longformqa')

from utils import extract_text_by_citation, citations_check


# %%
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

# %% [markdown]
# ### 1] Getting Started
# 
# We'll start by setting up the language model (LM) and retrieval model (RM). **DSPy** supports multiple API and local models. In this notebook, we'll work with GPT-3.5 (`gpt-3.5-turbo`) and the retriever `ColBERTv2`.
# 
# To make things easy, we've set up a ColBERTv2 server hosting a Wikipedia 2017 "abstracts" search index (i.e., containing first paragraph of each article from this [2017 dump](https://hotpotqa.github.io/wiki-readme.html)), so you don't need to worry about setting one up! It's free.
# 
# We configure **DSPy** to use the turbo LM and the ColBERTv2 retriever (over Wikipedia 2017 abstracts) by default. This can be overwritten for local parts of programs if needed.

# %%
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
turbo = dspy.OpenAI(model='gpt-4o-mini', max_tokens=500)
dspy.settings.configure(lm=turbo, trace=[], temperature=0.7)

# %% [markdown]
# ### 2] Dataset
# 
# Now, let's load a sample from the HotPotQA multi-hop dataset for our tasks. 

# %%
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0, keep_details=True)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

# %% [markdown]
# We just loaded `trainset` (300 examples) and `devset` (300 examples). Each example in our **training set** contains just a **question,** its corresponding (human-annotated) **answer**, and the **gold titles**. These gold titles represent titles of relevant Wikipedia articles that contain supporting facts necessary to answering the question. 
# 
# After loading the datasets, we'd applied `x.with_inputs('question')` to each example to tell **DSPy** that our input field in each example will be just `question`. Any other fields are labels not given to the system.
# 
# Now, let's look at some data examples.

# %%
train_example = trainset[0]
print(f"Question: {train_example.question}")
print(f"Answer: {train_example.answer}")
print(f"Relevant Wikipedia Titles: {train_example.gold_titles}")


# %%
dev_example = devset[18]
print(f"Question: {dev_example.question}")
print(f"Answer: {dev_example.answer}")
print(f"Relevant Wikipedia Titles: {dev_example.gold_titles}")

# %% [markdown]
# ### 3] LongFormQA with Citations
# 
# Let's define our first complete program for this task. We extend the `Multi-Hop QA` program, shifting the answer generation focus from short phrases of 1-5 words to comprehensive paragraphs that include citations. 
# 
# The `LongFormQA` module reflects the iterative multi-hop generation process in query generation, passage retrieval, and context assembly. The `GenerateCitedParagraph` layer then takes the context state alongside the question to generate a paragraph with relevant reference citations to the context. 
# %% [markdown]
# With this program, we aim to generate paragraphs that adhere the following guidelines:
# 1. Every 1-2 sentences in the paragraph are followed by citations in the intended format **"{text}... [source_num]."**
# 2. Every text segment preceding a citation is faithful to the referenced source passage. 

# %%
from dsp.utils import deduplicate

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateCitedParagraph(dspy.Signature):
    """Generate a paragraph with citations."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    paragraph = dspy.OutputField(desc="includes citations")

class LongFormQA(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_cited_paragraph(context=context, question=question)
        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        return pred

# %% [markdown]
# ### 4] Evaluation
# 
# We now define our evaluation metrics, **Intrinsic** and **Extrinsic** quality checks:
# 
# #### Intrinsic Metrics: passing internal computational constraints is the goal 
# 
# **Faithfulness (per Citation)**: To verify the accuracy of each citation in the generated text, we utilize another **DSPy** program: `ChainOfThought` of `CheckCitationFaithfulness`. This module takes segments of text preceding each citation and its corresponding context passage and determines whether the text accurately reflects the facts of the context. This validation process involves a language model call for each citation, ensuring that each reference in the generated paragraph is factually consistent with its reference source.

# %%
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context = dspy.InputField(desc="may contain relevant facts")
    text = dspy.InputField(desc="between 1 to 2 sentences")
    faithfulness = dspy.OutputField(desc="boolean indicating if text is faithful to context")

def citation_faithfulness(example, pred, trace):
    paragraph, context = pred.paragraph, pred.context
    citation_dict = extract_text_by_citation(paragraph)
    if not citation_dict:
        return False, None
    context_dict = {str(i): context[i].split(' | ')[1] for i in range(len(context))}
    faithfulness_results = []
    unfaithful_citations = []
    check_citation_faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    for citation_num, texts in citation_dict.items():
        if citation_num not in context_dict:
            continue
        current_context = context_dict[citation_num]
        for text in texts:
            try:
                result = check_citation_faithfulness(context=current_context, text=text)
                is_faithful = result.faithfulness.lower() == 'true'
                faithfulness_results.append(is_faithful)
                if not is_faithful:
                    unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'context': current_context})
            except ValueError as e:
                faithfulness_results.append(False)
                unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'error': str(e)})
    final_faithfulness = all(faithfulness_results)
    if not faithfulness_results:
        return False, None
    return final_faithfulness, unfaithful_citations

# %% [markdown]
# #### Extrinsic Metrics: Assess the overall quality and effectiveness of generated output on downstream task:
# 
# - **Citation Precision**: Measures proportion of cited 'gold titles' in generated paragraph from all cited titles for datapoint. 
# - **Citation Recall**: Measures proportion of cited 'gold titles' in generated paragraph from all 'gold titles' for datapoint.
# - **Answer Inclusion**: Evaluates whether generated paragraph with citations accurately incorporates the 'gold' answer for datapoint. 
# 
# 

# %%
def extract_cited_titles_from_paragraph(paragraph, context):
    cited_indices = [int(m.group(1)) for m in re.finditer(r'\[(\d+)\]\.', paragraph)]
    cited_indices = [index - 1 for index in cited_indices if index <= len(context)]
    cited_titles = [context[index].split(' | ')[0] for index in cited_indices]
    return cited_titles

def calculate_recall(example, pred, trace=None):
    gold_titles = set(example['gold_titles'])
    found_cited_titles = set(extract_cited_titles_from_paragraph(pred.paragraph, pred.context))
    intersection = gold_titles.intersection(found_cited_titles)
    recall = len(intersection) / len(gold_titles) if gold_titles else 0
    return recall

def calculate_precision(example, pred, trace=None):
    gold_titles = set(example['gold_titles'])
    found_cited_titles = set(extract_cited_titles_from_paragraph(pred.paragraph, pred.context))
    intersection = gold_titles.intersection(found_cited_titles)
    precision = len(intersection) / len(found_cited_titles) if found_cited_titles else 0
    return precision

def answer_correctness(example, pred, trace=None):
    assert hasattr(example, 'answer'), "Example does not have 'answer'."
    normalized_context = normalize_text(pred.paragraph)
    if isinstance(example.answer, str):
        gold_answers = [example.answer]
    elif isinstance(example.answer, list):
        gold_answers = example.answer
    else:
        raise ValueError("'example.answer' is not string or list.")
    return 1 if any(normalize_text(answer) in normalized_context for answer in gold_answers) else 0

# %% [markdown]
# We now evaluate our program on these metrics over our devset.

# %%
def evaluate(module):
    correctness_values = []
    recall_values = []
    precision_values = []
    citation_faithfulness_values = []
    for i in range(len(devset)):
        example = devset[i]
        try:
            pred = module(question=example.question)
            correctness_values.append(answer_correctness(example, pred))            
            citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)
            citation_faithfulness_values.append(citation_faithfulness_score)
            recall = calculate_recall(example, pred)
            precision = calculate_precision(example, pred)
            recall_values.append(recall)
            precision_values.append(precision)
        except Exception as e:
            print(f"Failed generation with error: {e}")

    average_correctness = sum(correctness_values) / len(devset) if correctness_values else 0
    average_recall = sum(recall_values) / len(devset) if recall_values else 0
    average_precision = sum(precision_values) / len(devset) if precision_values else 0
    average_citation_faithfulness = sum(citation_faithfulness_values) / len(devset) if citation_faithfulness_values else 0

    print(f"Average Correctness: {average_correctness}")
    print(f"Average Recall: {average_recall}")
    print(f"Average Precision: {average_precision}")
    print(f"Average Citation Faithfulness: {average_citation_faithfulness}")


# %%
longformqa = LongFormQA()
evaluate(longformqa)

# %% [markdown]
# Let's take a look at an example paragraph generation:

# %%
question = devset[15].question
pred = longformqa(question)
citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)

print(f"Question: {question}")
print(f"Predicted Paragraph: {pred.paragraph}")
print(f"Citation Faithfulness: {citation_faithfulness_score}")

# %% [markdown]
# We can see that the generated paragraph does not properly include citations as intended in the format of "[source]". 
# 
# Additionally, we see that not all included citations are faithful to their preceding text.
# %% [markdown]
# ### 5] Introducing Assertions: LongFormQAWithAssertions
# 
# To correct these errors, we introduce **Assertions** to impose clear computational constraints within our program.
# 
# DSPy provides two key mechanisms for **Assertions**:
# 
# - **`dspy.Assert`**: This mandates that the program must satisfy the given assertion, raising an Exception otherwise. This is important when enforcing non-negotiable constraints within the program.
# - **`dspy.Suggest`**: Unlike `Assert`, `Suggest` is more flexible. It encourages the program to meet the assertion but allows the program to continue even if the assertion is not satisfied. This is particularly useful for guiding the program towards desired outcomes without halting execution for non-critical issues.
# 
# Since our goal is indeed to evaluate the program on the defined metrics, let's utilize the `dspy.Suggest` assertion. 
# 
# The syntax for `dspy.Suggest` is as follows:
# ```python
# dspy.Suggest(validation_function(model_outputs): bool, instruction_message: str)
# ```
# 
# Let's add assertions to abide by the computational constraints defined above. 

# %%
class LongFormQAWithAssertions(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_cited_paragraph(context=context, question=question)
        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        dspy.Suggest(citations_check(pred.paragraph), "Make sure every 1-2 sentences has citations. If any 1-2 sentences lack citations, add them in 'text... [x].' format.", target_module=self.generate_cited_paragraph)
        _, unfaithful_outputs = citation_faithfulness(None, pred, None)
        if unfaithful_outputs:
            unfaithful_pairs = [(output['text'], output['context']) for output in unfaithful_outputs]
            for _, context in unfaithful_pairs:
                dspy.Suggest(len(unfaithful_pairs) == 0, f"Make sure your output is based on the following context: '{context}'.", target_module=self.generate_cited_paragraph)
        else:
            return pred
        return pred

# %% [markdown]
# We include assertions that simply reiterate our computational constraints and now allow the `LongFormQA` program to execute and adhere to these guidelines under the hood. 
# 
# Since we want to impose these assertions on the paragraph generation, we can pass in the `GenerateCitedParagraph` signature to indicate the `target_module` for the assertion handling to identify. 
# 
# In the first **Assertion**, we validate the output paragraph to ensure citations are included every 1-2 sentences. If this validation returns False, the assertion backtracking logic is activated the feedback instruction: **"Ensure each 1-2 sentences include citations in 'text... [x].' format."**
# 
# In the second **Assertion**, we now utilize the `CheckCitationFaithfulness` program to validate the accuracy of each cited references, looping over text segments denoted in the generated paragraph. In cases of unfaithful citations, it sends the feedback instruction alongside the context as: **"Ensure your output aligns with this context: '{context}'."** This ensures the assertion backtracking has the relevant information and specific context it needs.
# %% [markdown]
# Let's now evaluate our `LongFormQAWithAssertions` program over the devset.
# 
# Note that this requires wrapping the module with the `Retry` module which handles the backtracking logic. This wrapped module is then passed to the `assert_transform_module` function to prepare and execute the backtracking logic. This is passed alongside the `backtrack_handler` which configures the backtracking logic to account for the feedback messages passed in to the `dspy.Suggest` statements within the program.

# %%
longformqa_with_assertions = assert_transform_module(LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler) 
evaluate(longformqa_with_assertions)

# %% [markdown]
# Let's take a look at the same example from above with the `LongFormQAWithAssertions` program:

# %%
question = devset[15].question
pred = longformqa_with_assertions(question)
citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)

print(f"Question: {question}")
print(f"Predicted Paragraph: {pred.paragraph}")
print(f"Citation Faithfulness: {citation_faithfulness_score}")

# %% [markdown]
# We now see that both computational constraints are indeed met. Every 1-2 sentences includes a citation and from our `citation_faithfulness` check, we see that each reference is also faithful to its preceding text. 
# %% [markdown]
# ### 6] Compilation With Assertions
# 
# We can also leverage **DSPy**'s advanced compiling features to enhance our program's performance. 
# 
# For this, we utilize the `BootstrapFewShotWithRandomSearch` teleprompter, which automatically incorporates few-shot demonstrations and conducts a random search over a candidate set to output the best compiled program. We evaluate this over the `answer_correctness` metric as our ultimate goal is indeed to generate correct answers to the `HotPotQA` questions from the paragraphs, aiming to optimize both intrinsic and extrinsic metrics as a result. 
# 
# Let's evaluate this on the LongFormQA program first:

# %%
longformqa = LongFormQA()
teleprompter = BootstrapFewShotWithRandomSearch(metric = answer_correctness, max_bootstrapped_demos=2, num_candidate_programs=6)
cited_longformqa = teleprompter.compile(student = longformqa, teacher = longformqa, trainset=trainset, valset=devset[:25])
evaluate(cited_longformqa)

# %% [markdown]
# Let's evaluate this with assertions. 
# 
# **Note** The pipeline here lies in compiling with **Assertions** to give the teleprompter correct bootstrapped examples by the `answer_correctness` metric and then 'teaching' the student with these correct examples. This is represented by passing `LongFormQA()` as the student and `LongFormQAWithAssertions()` as the teacher.

# %%
longformqa = LongFormQA()
teleprompter = BootstrapFewShotWithRandomSearch(metric = answer_correctness, max_bootstrapped_demos=2, num_candidate_programs=6)
cited_longformqa_teacher = teleprompter.compile(student=longformqa, teacher = assert_transform_module(LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler), trainset=trainset, valset=devset[:25])
evaluate(cited_longformqa_teacher)

# %% [markdown]
# **Note** This pipeline on the other hand sets both the teacher and student with `LongFormQAWithAssertions()` to ensure the teacher correctly instructs the student with the right bootstrapped examples and the student has the chance to self-correct with **Assertions** for any examples that are still deemed incorrect.

# %%
longformqa = LongFormQA()
teleprompter = BootstrapFewShotWithRandomSearch(metric = answer_correctness, max_bootstrapped_demos=2, num_candidate_programs=6)
cited_longformqa_student_teacher = teleprompter.compile(student=assert_transform_module(LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler), teacher = assert_transform_module(LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler), trainset=trainset, valset=devset[:25])
evaluate(cited_longformqa_student_teacher)


