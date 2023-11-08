import hashlib
from typing import Optional
import streamlit as st
from langchain.callbacks.manager import collect_runs
from langchain.prompts import BasePromptTemplate
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from eval_suggestions import suggest_evals
from langchain import hub
from langchainhub import Client as HubClient
import asyncio
import functools
import base64

from rich import print


# Color mapping for each tag
color_mapping = {
    "FormatInstruction": "#FFCCCC",  # Light Red
    "ExampleDemonstration": "#FFFF99",  # Light Yellow
    "PromptRephrasing": "#CCFFFF",  # Light Cyan
    "WorkflowDescription": "#C8A2C8",  # Lilac
    "DataPlaceholders": "#CCCCFF",  # Light Periwinkle
    "QuantityInstruction": "#FFCC99",  # Light Orange
    "Inclusion": "#CCFF99",  # Light Lime
    "Exclusion": "#FF99CC",  # Light Pink
    "QualitativeAssessment": "#99CCFF",  # Light Blue Gray
}

ignore_tags = ["PromptRephrasing", "DataPlaceholders"]

tooltip_descriptors = {
    "FormatInstruction": "This type of delta specifies the desired format of the response, such as requiring it to be a Python dictionary.",
    "ExampleDemonstration": "This type of delta demonstrates the detailed structure and content expected from the response, providing clear patterns to follow.",
    "WorkflowDescription": "This type of delta outlines the step-by-step workflow the response should articulate, guiding the LLM through the process.",
    "QuantityInstruction": "This type of delta specifies a rule around a number of items in a response.",
    "Inclusion": "This type of delta specifies that some key words or phrases must be included in a response.",
    "Exclusion": "This type of delta specifies that some key words or phrases must be excluded from a response.",
    "QualitativeAssessment": "This type of delta introduces subjective or hard-to-evaluate criteria for a good response.",
}


def apply_color_styles(prompt_with_tags, color_mapping):
    for tag, color in color_mapping.items():
        if tag in ignore_tags:
            # Replace the tag with nothing
            prompt_with_tags = prompt_with_tags.replace(
                f"<{tag}>", ""
            ).replace(f"</{tag}>", "")

        else:
            prompt_with_tags = prompt_with_tags.replace(
                f"<{tag}>", f"<span style='background-color: {color};'>"
            ).replace(f"</{tag}>", "</span>")
    return prompt_with_tags


client = Client()
hub_client = HubClient()

st.set_page_config(
    page_title="SPADE: System for Prompt Analysis and Delta-based Evaluation",
    page_icon="🦜️️🛠️ ♠️",
)

st.subheader("🦜🛠️ ♠️ Get Suggested Evalution Functions")
st.write(
    "SPADE ♠️ (System for Prompt Analysis and Delta-based Evaluation) will suggest binary eval functions for your prompt that you can run on all future LLM responses on. It works best when given the version history of your prompt (i.e., through a [LangSmith Hub](https://smith.langchain.com/hub) repo), but you can also use it with a single prompt template."
)

# Include CTA
st.info(
    "This is an experimental version of SPADE, built in collaboration with UC Berkeley and used for research purposes. [Here's a link](https://blog.langchain.dev/p/660b0a4c-d89a-48fe-9e49-86f1132d536f/) to our blog post. If you'd like to provide feedback on the quality of evals or participate in an interactive prompt engineering study so we can improve the tool, please fill out [this form](https://forms.gle/ph3Y6nTZWhPn3w8W8)."
)

with st.expander("ℹ️ How it works"):
    st.write(
        "SPADE uses GPT-4 to first determine any additions to prompts between consecutive versions. If you don't have prompt version history, it compares your prompt to a blank prompt."
    )
    st.write(
        'Then, given the additions to a prompt (e.g., adding the phrase "limit your answer to 5 words"), it synthesizes a set of evaluation functions that can be used to evaluate future responses to your prompt (e.g., a function that splits the response into a list of words and checks that there are fewer than 5 elements).'
    )
    st.write(
        "Evaluation functions take in the response text and return a boolean. Most will be simple, but some are more complex and may require an LLM to run. SPADE will tell you which ones need an LLM, and the function will include a call to `ask_llm` (a placeholder function that asks a yes/no question an LLM)."
    )

# Initialize session state variables
if "versions" not in st.session_state:
    st.session_state.versions = []
if "eval_functions" not in st.session_state:
    st.session_state.eval_functions = []
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "diff_to_render" not in st.session_state:
    st.session_state.diff_to_render = None

feedback_option = "thumbs"
option = st.selectbox("Choose Input Type", ["Prompt Template", "Hub Repo"])


def list_versions(repo_full_name: str):
    hashes = [
        v["commit_hash"]
        for v in hub_client.list_commits(repo_full_name)["commits"]
    ]
    return [hash[:7] for hash in hashes]


def on_feedback_submit(feedback, run_ids):
    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            print("score", score)
            # Formulate feedback type string incorporating the feedback option
            # and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string
            # and optional comment
            for run_id in run_ids:
                feedback_record = client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                )
            print("Feedback recorded!", feedback_record)
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            st.warning("Invalid feedback score.")


if option == "Prompt Template":
    string1 = st.text_area(
        "Prompt Template",
        "",
        placeholder="A client ({client_genders}) wants to be styled for {event}. Suggest 5 apparel items for {client_pronoun} to wear. For wedding-related events, don’t suggest any white items unless the client explicitly states that they want to be styled for their wedding. Return your answer as a python list of strings",
    )
    if string1.strip():
        st.session_state.versions = ["", string1]
        st.session_state.source = "prompt_template"
else:
    string1 = st.text_input(
        "[LangSmith Hub Repo](https://smith.langchain.com/hub):",
        "",
        placeholder="flflo/summarization",
    )
    if string1.strip():
        version = ""
        if ":" in string1:
            string1, version = string1.split(":", 1)
        with st.spinner("Fetching prompt versions..."):
            hashes = list_versions(string1)
            if version:
                hashes_ = []
                for hash in hashes:
                    hashes_.append(hash)
                    if hash.startswith(version):
                        break
                hashes = hashes_
            try:
                prompts = []
                for hash_ in hashes:
                    prompts.append(hub.pull(string1 + ":" + hash_))
            except Exception as e:
                st.error(repr(e))
                exit()

            st.session_state.versions = [
                prompt.format(
                    **{k: f"{{{k}}}" for k in prompt.input_variables}
                )
                for prompt in prompts
            ]

            # Append empty prompt to the beginning
            st.session_state.versions = [""] + st.session_state.versions
            st.session_state.source = string1.strip()

if st.session_state.versions:
    # Display the versions
    with st.expander("Prompt Versions"):
        # Create table of versions with keys 0...n and values of versions
        versions_table = [
            {"prompt": st.session_state.versions[i]}
            for i in range(len(st.session_state.versions))
        ]
        st.table(versions_table)

    # Check if we need to run the main computation
    run_computation = (
        "eval_functions" not in st.session_state
        or not st.session_state.eval_functions
    )

    if run_computation:
        with collect_runs() as cb:

            def typing_callback(
                text: Optional[str], agg: list, message_placeholder
            ):
                if text is None:
                    # Erase
                    message_placeholder.markdown("")
                    return
                agg.append(text)
                message_placeholder.markdown("".join(agg) + "▌")

            progress_text = "Eval function generation progress..."
            progress_bar = st.progress(0.01, text=progress_text)

            with st.expander("See In-Progress Analysis", expanded=False):
                # Go through consecutive versions of the prompt
                idx = 0
                for version_1, version_2 in zip(
                    st.session_state.versions,
                    st.session_state.versions[1:],
                ):
                    st.markdown(
                        f"Comparing versions {idx} and {idx + 1} of the prompt..."
                    )
                    char_response = []
                    message_placeholder = st.empty()
                    function_response = []
                    eval_placeholder = st.empty()
                    (
                        eval_functions,
                        message_history,
                        diff_to_render,
                    ) = asyncio.run(
                        suggest_evals(
                            version_1,
                            version_2,
                            source=st.session_state.source,
                            characterize_callback=functools.partial(
                                typing_callback,
                                agg=char_response,
                                message_placeholder=message_placeholder,
                            ),
                            eval_prediction_callback=functools.partial(
                                typing_callback,
                                agg=function_response,
                                message_placeholder=eval_placeholder,
                            ),
                        )
                    )
                    message_placeholder.markdown("".join(char_response))
                    eval_placeholder.markdown("".join(function_response))

                    # Save results to session state
                    st.session_state.eval_functions += eval_functions
                    st.session_state.message_history += message_history
                    if not st.session_state.diff_to_render:
                        st.session_state.diff_to_render = diff_to_render

                    # Increment index
                    idx += 1
                    progress_bar.progress(
                        float(idx / len(st.session_state.versions[1:])),
                        text=progress_text,
                    )

                st.session_state.run_ids = [run.id for run in cb.traced_runs]


def log_download_event(run_ids):
    for run_id in run_ids:
        client.create_feedback(
            run_id,
            "download_button_clicked",
            score=1,
        )


if st.session_state.get("run_ids"):
    # Show diff
    if st.session_state.diff_to_render:
        try:
            diff_to_render = apply_color_styles(
                st.session_state.diff_to_render, color_mapping
            )

            st_cols = st.columns(2)

            st_cols[0].write("#### Annotated first prompt template")
            st_cols[0].markdown(diff_to_render, unsafe_allow_html=True)
            # Create and display the legend for color mapping
            st_cols[1].write("#### Prompt refinement legend")
            for tag, color in color_mapping.items():
                if tag not in ignore_tags:
                    st_cols[1].markdown(
                        f"<span style='display:inline-block;width:12px;height:12px;background-color:{color};'></span> {tag}",
                        unsafe_allow_html=True,
                        help=tooltip_descriptors.get(tag),
                    )
        except Exception as e:
            st.error(f"An error occurred while rendering your prompt: {e}")

    # Show eval functions
    st.write("#### ♠️ Suggested evaluation functions")
    all_functions = [
        "import json\nimport re\nimport numpy as np\nimport pandas as pd"
    ]
    for i, function in enumerate(st.session_state.eval_functions):
        code_id = (
            "code_"
            + hashlib.md5(str(function["code"]).encode()).hexdigest()[:6]
        )
        code_html = f"""
<pre><code id="{code_id}" class="language-python"># Needs LLM: {function['needs_llm']}
{function['code']}
</code></pre>
"""
        all_functions.append(
            f"# Needs LLM: {function['needs_llm']}\n{function['code']}"
        )
        st.markdown(code_html, unsafe_allow_html=True)

    # Create a Streamlit button to download the Python file
    st.download_button(
        label="Download Functions as Python File",
        data="\n\n".join(all_functions),
        file_name="eval_functions.py",
        mime="text/plain",
        type="primary",
        on_click=lambda: log_download_event(st.session_state.run_ids),
    )

    # Show full message history
    with st.expander("GPT-4 message history", expanded=False):
        # Get the "assistant" messages
        st.write(
            "Here are the full GPT-4 responses to (1) categorize the prompt additions and (2) generate evaluation functions."
        )
        st.write(
            [
                m
                for m in st.session_state.message_history
                if m["role"] == "assistant"
            ]
        )

    run_ids = st.session_state.run_ids

    st.write("Help us improve! Did you find any of this output useful?")
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[Optional] Please provide an explanation",
        key=f"thumbs_feedback_{run_ids[0]}",
        on_submit=on_feedback_submit,
        args=[run_ids],
    )
