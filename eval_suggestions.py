"""
Module to suggest evals for the user to run based on prompt deltas.
"""
import difflib
import hashlib
import json
from typing import Callable, Optional

from copy import deepcopy
from dotenv import load_dotenv

import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

import logging
import re
from langchain.adapters.openai import convert_openai_messages
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.callbacks.manager import trace_as_chain_group

llm = ChatOpenAI(model="gpt-4")

load_dotenv()

import os

# BASE_URL = os.getenv("AZURE_API_BASE")
# API_KEY = os.getenv("AZURE_API_KEY")
# DEPLOYMENT_NAME = "gpt-4"
# llm = AzureChatOpenAI(
#     openai_api_base=BASE_URL,
#     openai_api_version=os.getenv("AZURE_API_VERSION"),
#     deployment_name=DEPLOYMENT_NAME,
#     openai_api_key=API_KEY,
#     openai_api_type="azure",
# )

CONCEPT_TEMPLATE = """Here is the diff for my prompt template:

"{prompt_diff}"

Based on only the changed lines, I want to write assertions for my LLM pipeline to run on all pipeline responses. Here are some categories of assertion concepts I want to check for:

- PresentationFormat: Is there a specific format for the response, like a comma-separated list or a JSON object?
- ExampleDemonstration: Does theh prompt template include any examples of good responses that demonstrate any specific headers, keys, or structures?
- WorkflowDescription: Does the prompt template include any descriptions of the workflow that the LLM should follow, indicating possible assertion concepts?
- QuantityInstruction: Are there any instructions regarding the number of items of a certain type in the response, such as “at least”, “at most”, or an exact number?
- Inclusion: Are there keywords that every LLM response should include?
- Exclusion: Are there keywords that every LLM response should never mention?
- QualitativeAssessment: Are there qualitative criteria for assessing good responses, including specific requirements for length, tone, or style?
- Other: Based on the prompt template, are there any other concepts to check in assertions that are not covered by the above categories, such as correctness, completeness, or consistency?

Give me a list of concepts to check for in LLM responses. Each item in the list should contain a string description of a concept to check for, its corresponding category, and the source, or phrase in the prompt template that triggered the concept. For example, if the prompt template is "I am a still-life artist. Give me a bulleted list of colors that I can use to paint <object>.", then a concept might be "The response should include a bulleted list of colors." with category "PresentationFormat" and source "Give me a bulleted list of colors".

Your answer should be a JSON list of objects within ```json ``` markers, where each object has the following fields: "concept", "category", and "source". This list should contain as many assertion concepts as you can think of, as long are specific and reasonable."""


def show_diff(template_1: str, template_2: str):
    # Split the templates into sentences
    if isinstance(template_1, list):
        template_1 = str(template_1)
    if isinstance(template_2, list):
        template_2 = str(template_2)

    sent_1 = sent_tokenize(template_1)
    sent_2 = sent_tokenize(template_2)

    diff = list(difflib.unified_diff(sent_1, sent_2))

    # Convert diff to string
    diff = "\n".join(diff)
    return diff


FUNCTION_GEN_TEMPLATE = """Here is my prompt template:

"{prompt_template}"

Here is an example and its corresponding LLM response:

Example formatted LLM prompt: {example_prompt}
LLM Response: {example_response}

Here are the concepts I want to check for in LLM responses:

{concepts}

Give me a list of assertions as Python functions that can be used to check for these concepts in LLM responses. Assertion functions should not be decomposed into helper functions. Assertion functions can leverage the external function `ask_llm` if the concept is too hard to evaluate with Python code alone (e.g., qualitative criteria). The `ask_llm` function accepts formatted_prompt, response, and question arguments and submits this context to an expert LLM, which returns True or False based on the context. Since `ask_llm` calls can be expensive, you can batch similar concepts that require LLMs to evaluate into a single assertion function, but do not cover more than two concepts with a function. For concepts that are ambiguous to evaluate, you should write multiple different assertion functions (e.g., different `ask_llm` prompts) for the same concept(s). Each assertion function should have no more than one `ask_llm` call.

Each function should take in 3 args: an example (dict with string keys), prompt formatted on that example (string), and LLM response (string). Each function shold return a boolean indicating whether the response satisfies the concept(s) covered by the function. Here is a sample assertion function for an LLM pipeline that generates summaries:

```python
async def assert_simple_and_coherent_narrative(example: dict, prompt: str, response: str):
    # Check that the summary form a simple, coherent narrative telling a complete story.

    question = "Does the summary form a simple, coherent narrative telling a complete story?"
    return await ask_llm(prompt, response, question)
```

Your assertion functions should be distinctly and descriptively named, and they should include a short docstring describing what the function is checking for. All functions should be asynchronous and use the `await` keyword when calling `ask_llm`."""

RENDER_DIFF_TEMPLATE = """Please use this JSON structure, detailing the newly added instructions to the LLM prompt template, to render the second prompt template with the changes highlighted. You should return the same second prompt template, but wrap each identified change based on the JSON structure of changes in its tag. Make sure each change has opening and closing tags (e.g., <FormatInstruction></FormatInstruction>). Category tags should not be nested. Your answer should start with <FormattedPromptTemplate> and end with </FormattedPromptTemplate>"""

# Hash all prompts into a single string
combined_content = (
    CONCEPT_TEMPLATE + FUNCTION_GEN_TEMPLATE + RENDER_DIFF_TEMPLATE
).encode()
hash_object = hashlib.sha256(combined_content)
PIPELINE_PROMPT_HASH = hash_object.hexdigest()


async def suggest_evals(
    template_1: str,
    template_2: str,
    source: str,
    characterize_callback: Optional[Callable[[str], None]] = None,
    eval_prediction_callback: Optional[Callable[[str], None]] = None,
):
    """Suggest evals for the user to run based on prompt deltas."""
    with trace_as_chain_group(
        "suggest_evals",
        inputs={
            "template_1": template_1,
            "template_2": template_2,
        },
        tags=[source, PIPELINE_PROMPT_HASH],
    ) as cb:
        # If the templates are the same, return []
        if template_1 == template_2:
            # Send callback that there is no change
            characterize_callback("Prompt templates are the same.")

            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": [],
                    "rendered_diff": None,
                }
            )
            return [], [], None

        diff = show_diff(template_1, template_2)
        messages = [
            {
                "content": "You are an expert in Python and prompting large language models (LLMs). You are assisting me, a prompt engineer, build and monitor an LLM pipeline. An LLM pipeline accepts a prompt, with some instructions, and uses an LLM to generate a response to the prompt. A prompt engineer writes a prompt template, with placeholders, that will get formatted with different variables at pipeline runtime. Typically as prompt engineers test a pipeline, we observe that some responses are not good, so we add new instructions to the prompt template to prevent against these failure modes.",
                "role": "system",
            },
            {
                "content": CONCEPT_TEMPLATE.format(
                    prompt_diff=diff,
                ),
                "role": "user",
            },
        ]

        # First characterize the deltas
        try:
            lc_messages = convert_openai_messages(messages)
            char_response = llm.astream(lc_messages, {"callbacks": cb})

            logging.debug("Determining prompt deltas...")
            collected_messages = []
            # iterate through the stream of events
            async for chunk in char_response:
                if characterize_callback:
                    characterize_callback(chunk.content)
                collected_messages.append(chunk.content)

            logging.info("")
            reply = "".join(collected_messages)

        except Exception as e:
            logging.error(f"Error getting deltas: {e}")
            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": messages,
                    "rendered_diff": None,
                }
            )
            return [], messages, None

        # Parse the reply's json from ```json ... ```
        concepts = None
        try:
            reply = re.search(r"```json(.*?)\n```", reply, re.DOTALL).group(1)
            concepts = json.loads(reply)
        except Exception as e:
            logging.error(f"Error parsing json: {e}")
            messages.append({"content": reply, "role": "assistant"})
            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": messages,
                    "rendered_diff": None,
                }
            )
            return [], messages, None

        # Render the diff
        try:
            diff_render_messages = deepcopy(messages)
            diff_render_messages.append(
                {"content": RENDER_DIFF_TEMPLATE, "role": "user"}
            )
            diff_render_response = await llm.ainvoke(
                convert_openai_messages(diff_render_messages)
            )
            diff_render_response = diff_render_response.content

            # Extract whatever is in ```<FormattedPromptTemplate> tags
            pattern = r"<FormattedPromptTemplate>(.*?)</FormattedPromptTemplate>"
            matches = re.findall(pattern, diff_render_response, re.DOTALL)
            diff_render_response = matches[0]

        except Exception as e:
            logging.error(f"Error rendering diff: {e}")
            diff_render_response = None

        # If there are no changes, return []
        if not concepts:
            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": messages,
                    "rendered_diff": diff_render_response,
                }
            )
            return [], messages, diff_render_response

        # Create sample prompt and response by querying GPT-4
        example_prompt_llm_response = await llm.ainvoke(
            convert_openai_messages(
                [
                    {
                        "content": "You are a helpful synthetic data creator.",
                        "role": "system",
                    },
                    {
                        "content": f"Here is my prompt template:\n\n{template_2}\n\nPlease return a JSON object where the keys are variables in the template and the values are realistic. Cover all the variables. Return your answer within ```json ``` markers.",
                        "role": "user",
                    },
                ]
            )
        )
        reply = example_prompt_llm_response.content

        try:
            reply = re.search(r"```json(.*?)\n```", reply, re.DOTALL).group(1)
            sample_example = json.loads(reply)

            # Get formatted prompt
            formatted_prompt = template_2.format(**sample_example)
        except Exception as e:
            logging.error(f"Error parsing sample example json: {e}")
            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": messages,
                    "rendered_diff": None,
                }
            )
            return [], messages, diff_render_response

        # Get response for formatted prompt
        example_response_llm_response = await llm.ainvoke(
            convert_openai_messages(
                [
                    {
                        "content": formatted_prompt,
                        "role": "user",
                    },
                ]
            )
        )
        example_response = example_response_llm_response.content

        # Construct prompt to LLM
        function_generation_messages = [
            {
                "content": "You are an expert Python programmer and helping me write assertions for my LLM pipeline. An LLM pipeline accepts an example and prompt template, fills the template's placeholders with the example, and generates a response.",
                "role": "system",
            },
            {
                "content": FUNCTION_GEN_TEMPLATE.format(
                    prompt_template=template_2,
                    example_prompt=formatted_prompt,
                    example_response=example_response,
                    concepts=concepts,
                ),
                "role": "user",
            },
        ]

        # See if there are any deltas that bring upon evals
        eval_functions = []
        logging.debug("Generating evals...")
        lc_messages = convert_openai_messages(function_generation_messages)
        eval_response_stream = llm.astream(lc_messages, {"callbacks": cb})
        eval_response = []
        async for chunk in eval_response_stream:
            if eval_prediction_callback:
                eval_prediction_callback(chunk.content)
            eval_response.append(chunk.content)
        eval_prediction_callback(None)
        eval_response_content = "".join(eval_response)
        messages.append({"content": eval_response_content, "role": "assistant"})

        # Extract all functions within ```python ``` markers
        eval_functions = re.findall(
            r"```python(.*?)\n```", eval_response_content, re.DOTALL
        )
        eval_functions = [f.strip() for f in eval_functions]
        eval_functions = [f for f in eval_functions if f != ""]
        cb.on_chain_end(
            {
                "eval_functions": eval_functions,
                "messages": messages,
                "rendered_diff": diff_render_response,
            },
        )
        return eval_functions, messages, diff_render_response


if __name__ == "__main__":
    import asyncio

    ef, m = asyncio.run(
        suggest_evals(
            "You ar ea helpful agent",
            "You are a helpful agent. Respond in less than 3 words",
        )
    )
    print(ef, m)
