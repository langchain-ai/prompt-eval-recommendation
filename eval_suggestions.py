"""
Module to suggest evals for the user to run based on prompt deltas.
"""
import difflib
import json
from typing import Callable, Optional


from dotenv import load_dotenv

import logging
import re
from langchain.adapters.openai import convert_openai_messages
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import trace_as_chain_group

llm = ChatOpenAI(model="gpt-4")


load_dotenv()


def show_diff(template_1: str, template_2: str):
    diff = difflib.ndiff(template_1.splitlines(), template_2.splitlines())
    return "\n".join(diff)


def show_readable_diff(template_1: str, template_2: str):
    diff = list(
        difflib.ndiff(template_1.splitlines(), template_2.splitlines())
    )

    # Processing the diff list for readable format
    added_lines = []
    deleted_lines = []

    for line in diff:
        if line.startswith("- "):
            deleted_lines.append(line)
        if line.startswith("+ "):
            added_lines.append(line)

    output = []

    if deleted_lines:
        output.append("Deleted lines:")
        output.extend(deleted_lines)

    if added_lines:
        output.append("Added lines:")
        output.extend(added_lines)

    return "\n".join(output)


CATEGORIZE_TEMPLATE = """I need your assistance in identifying any new instructions I've added to my prompt. The new instructions will help inform the creation of new evaluation functions for future responses to my prompt.

**Prompt Template 1**: 
```
{template_1}
```

**Prompt Template 2**:
```
{template_2}
```

**Additions Highlighted**:
```
{diff}
```

First, analyze the given templates and categorize the new additions to the prompt according to the following criteria:

- **Structural**:
  - **Presentation Format**: Did I add any specification of the expected response format, such as a list, JSON, Markdown, or HTML?
  - **Example Demonstration**: Did I add any examples that demonstrate the desired response beyond the format, such as specific headers or keys?
  - **Prompt Rephrasing (not a new instruction)**: Did I rephrase the prompt to clarify the task?
  - **Data Sources or Context Addition (not a new instruction)**: Did I add any new data sources or context, like variables or placeholders, to help the LLM give a better response?

- **Content**:
  - **Count**: Did I add any new specifications regarding the number of items of some type in the response, like "at least", "at most", or an exact count?
  - **Inclusion**: Did I add any new phrases or ideas to the prompt that every future LLM response should include?
  - **Exclusion**: Did I specify any new phrases or ideas that every future LLM response should exclude?
  - **Scorecard Items**: Did I add any new qualitative criteria of good responses, such as a specific length, tone, or style?

**Expected Output Structure**:

```json
{{
  "Structural": {{
    "PresentationFormat": "Describe the new high-level format (if any)",
    "ExampleDemonstration": "Describe the new structure every response should follow (if any)",
    "PromptRephrasing": "Change description (if any)",
    "DataOrContextAddition": "Change description (if any)"
  }},
  "Content": {{
    "Count": "Describe the new item that should be counted (if any)",
    "Inclusion": "State the new phrase that should be included in every response (if any)",
    "Exclusion": "State the new phrase that should be excluded from every response (if any)",
    "ScorecardItems": "Describe the new qualitative criteria of good responses (if any)"
  }}
}}
```

Fill out the structure based on your analysis. For categories without changes, indicate "No change"."""

SUGGEST_EVAL_TEMPLATE = """Using the JSON output detailing the new instructions I added to the LLM prompt, design Python functions to run on all future responses to my prompt #2 above that makes sure the new instructions are followed. Do not suggest any evaluation functions for PromptRephrasing and DataOrContextAddition modifications. The functions should operate on LLM responses and adhere to the following requirements:

1. Only use the `json`, `re`, and standard Python libraries.
2. For complex evaluations that can't be handled purely by pattern matching or logical checks, you may use the ask_expert function. This function sends a specific yes-or-no question to a human expert and returns a boolean value. Use this sparingly, as it is expensive.
3. Evaluation functions should return a binary True or False value.
4. For scorecard items, avoid generic evaluations. Instead, explicitly evaluate the response against the specific criteria provided. For instance, if the scorecard item specifies a "concise response", the function might check the length of the response and decide whether it's concise.

**Function Signature**:
```python
def evaluation_function_name(response: str) -> bool:
    # Your implementation here
```

**Given the JSON Output of Changes**:
We want to derive automated evaluation functions to verify if the responses adhere to the requirements.

**Example Functions for Each Type of Change**:

{example_evals}

Keep in mind that these functions serve as templates. You'll need to adjust the specifics (like the exact phrases or counts) based on the actual criteria I've added to my prompts. Each function should be complete a standalone validation task to run on every response to my prompt. Each function should be a correct Python function that I can run on every LLM response for all future pipeline runs."""

EXAMPLE_EVALS = {
    "PresentationFormat": """**Presentation Format**:
    - If the desired format is JSON:
    ```python
    def evaluate_json_format(response: str) -> bool:
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False
    ```
    - If the desired format is a list:
    ```python
    def evaluate_list_format(response: str) -> bool:
        # Check if response starts with a bullet point or number
        return response.startswith("- ") or response.startswith("1. ")
    ```
    """,
    "ExampleDemonstration": """**Example Demonstration**:
    ```python
    def check_example_demonstration(response: str) -> bool:
        # Suppose the example is to start the response with "In my opinion,"
        if response.startswith("In my opinion,"):
            return True
        return False
    ```
    """,
    "Count": """**Count**:
    ```python
    def evaluate_num_distinct_words(response: str) -> bool:
        # Suppose responses should contain at least 3 distinct words
        distinct_word_count = len(set(response.split()))
        return distinct_word_count >= 3
    ```
    """,
    "Inclusion": """**Inclusion**:
    ```python
    def check_includes_color(response: str) -> bool:
        # Suppose the response should include some color
        colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "navy"]
    
        return any(color in response for color in colors)
    ```
    """,
    "Exclusion": """**Exclusion**:
    ```python
    def check_excludes_phrases(response: str) -> bool:
        forbidden_phrases = ["forbidden phrase 1", "forbidden phrase 2"]
        return not any(phrase in response for phrase in forbidden_phrases)
    ```
    """,
    "ScorecardItems": """**Scorecard Items**:
    - If the desired length is concise:
    ```python
    def evaluate_concise(response: str) -> bool:
        # Suppose the response should be less than 50 characters
        return len(response) < 50
    ```
    - If the desired tone is positive:
    ```python
    def evaluate_tone(response: str) -> bool:
        return ask_expert("Is the tone of the response positive?")
    ```
    """,
}


def get_suggest_eval_prompt(changes_flagged):
    # See which keys have been flagged
    examples_to_include = [
        f"{str(i)}. {EXAMPLE_EVALS[key]}"
        for i, key in enumerate(changes_flagged)
        if key in EXAMPLE_EVALS
    ]
    # Format the prompt
    examples_to_include_str = "\n".join(examples_to_include)
    return SUGGEST_EVAL_TEMPLATE.format(example_evals=examples_to_include_str)


async def suggest_evals(
    template_1: str,
    template_2: str,
    characterize_callback: Optional[Callable[[str], None]] = None,
    eval_prediction_callback: Optional[Callable[[str], None]] = None,
):
    """Suggest evals for the user to run based on prompt deltas."""
    with trace_as_chain_group(
        "suggest_evals",
        inputs={"template_1": template_1, "template_2": template_2},
    ) as cb:
        # If the templates are the same, return []
        if template_1 == template_2:
            cb.on_chain_end({"eval_functions": [], "messages": []})
            return [], []

        template_1_pretty = template_1 if template_1 != "" else "Empty string"
        template_2_pretty = template_2 if template_2 != "" else "Empty string"

        diff = show_readable_diff(template_1, template_2)
        messages = [
            {
                "content": "You are a Python expert and expert in prompting large language models (LLMs). You are assisting me write evaluation functions for a complex LLM pipeline. An LLM pipeline accepts a prompt, with some instructions, and creates a response to answer the prompt. Typically as a developer tests a pipeline, they observe that some responses are not good, so they add new instructions to their prompt to prevent against these failure modes. Based on edits I make to a prompt template (which are indicative of possible failure modes), you will infer some evaluation functions that I should run on future LLM pipeline responses.",
                "role": "system",
            },
            {
                "content": CATEGORIZE_TEMPLATE.format(
                    template_1=template_1_pretty,
                    template_2=template_2_pretty,
                    diff=diff,
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
            cb.on_chain_end({"eval_functions": [], "messages": messages})
            return [], messages

        # Parse the reply's json from ```json ... ```
        reply_json = None
        try:
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, reply, re.DOTALL)
            reply_json = json.loads(matches[0])
        except Exception as e:
            logging.error(f"Error parsing json: {e}")
            messages.append({"content": reply, "role": "assistant"})
            cb.on_chain_end({"eval_functions": [], "messages": messages})
            return [], messages

        # Look for any changes
        changes_made = []
        for key in reply_json.get("Structural", {}):
            if not isinstance(reply_json["Structural"][key], str):
                continue

            if reply_json["Structural"][key].lower() != "no change":
                changes_made.append(key)
        for key in reply_json.get("Content", {}):
            if not isinstance(reply_json["Content"][key], str):
                continue

            if reply_json["Content"][key].lower() != "no change":
                changes_made.append(key)

        # Remove promptrephrasing and dataorcontextaddition
        if "PromptRephrasing" in changes_made:
            changes_made.remove("PromptRephrasing")
        if "DataOrContextAddition" in changes_made:
            changes_made.remove("DataOrContextAddition")

        # If there are no changes, return []
        if not changes_made:
            cb.on_chain_end({"eval_functions": [], "messages": messages})
            return [], messages

        # Delete the PromptRephrasing and DataOrContextAddition keys
        if "PromptRephrasing" in reply_json["Structural"]:
            del reply_json["Structural"]["PromptRephrasing"]
        if "DataOrContextAddition" in reply_json["Structural"]:
            del reply_json["Structural"]["DataOrContextAddition"]

        messages.append(
            {
                "content": reply,
                "role": "assistant",
            }
        )

        # See if there are any deltas that bring upon evals
        eval_functions = []
        # Then extract the evals
        messages.append(
            {"content": get_suggest_eval_prompt(changes_made), "role": "user"}
        )
        logging.debug("Generating evals...")
        lc_messages = convert_openai_messages(messages)
        eval_response_stream = llm.astream(lc_messages, {"callbacks": cb})
        eval_response = []
        async for chunk in eval_response_stream:
            if eval_prediction_callback:
                eval_prediction_callback(chunk.content)
            eval_response.append(chunk.content)
        eval_prediction_callback(None)
        eval_response_content = "".join(eval_response)
        messages.append(
            {"content": eval_response_content, "role": "assistant"}
        )

        # Look for the evals in the response as any instance of ```python ```
        # pattern = r"```python(.*?)```"
        # pattern = r"```python\s+(.*?def.*?)(?=\n```)"  # match any def
        pattern = r"^\s*```python\s+(.*?def.*?)(?=\n\s*```)"  # match any def with leading whitespace
        matches = re.findall(
            pattern, eval_response_content, re.DOTALL | re.MULTILINE
        )

        # Get longest match
        for match in matches:
            match = match.strip()

            try:
                # Replace `ask_expert` with a call to an llm function
                needs_llm = False
                function_str = match
                if "ask_expert" in function_str:
                    function_str = function_str.replace(
                        "ask_expert", "ask_llm"
                    )
                    needs_llm = True

                # Add the function to the list of eval functions
                eval_functions.append(
                    {"code": function_str, "needs_llm": needs_llm}
                )

            except:
                logging.error(f"Error parsing code: {match}")
        cb.on_chain_end(
            {"eval_functions": eval_functions, "messages": messages}
        )
        return eval_functions, messages


if __name__ == "__main__":
    import asyncio

    ef, m = asyncio.run(
        suggest_evals(
            "You ar ea helpful agent",
            "You are a helpful agent. Respond in less than 3 words",
        )
    )
    print(ef, m)
