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

from rich import print

# TODOS:
# Log interactions with the copy code button -- this is a bit tricky because
# we'd need to make a javascript code rendering component with a copy button

client = Client()
hub_client = HubClient()

st.set_page_config(
    page_title="Capturing User Feedback",
    page_icon="🦜️️🛠️",
)

st.subheader("🦜🛠️ Get Suggested Evalution Functions")
st.write(
    "This tool will suggest binary eval functions for your prompt that you can run on all future LLM responses on. It works best when given the version history of your prompt (i.e., through a LangSmith Hub repo), but you can also use it with a single prompt template."
)

with st.expander("ℹ️ How it works"):
    st.write(
        "This tool uses GPT-4 to first determine any additions to prompts between consecutive versions. If you don't have prompt version history, it compares your prompt to a blank prompt."
    )
    st.write(
        'Then, given the additions to a prompt (e.g., adding the phrase "limit your answer to 5 words"), it synthesizes a set of evaluation functions that can be used to evaluate future responses to your prompt (e.g., a function that splits the response into a list of words and checks that there are fewer than 5 elements).'
    )
    st.write(
        "Evaluation functions take in the response text and return a boolean. Most will be simple, but some are more complex and may require an LLM to run. The tool will tell you which ones need an LLM, and the function will include a call to `ask_llm` (a placeholder function that asks a yes/to question an LLM)."
    )

# Initialize session state variables
if "versions" not in st.session_state:
    st.session_state.versions = []
if "eval_functions" not in st.session_state:
    st.session_state.eval_functions = []
if "message_history" not in st.session_state:
    st.session_state.message_history = []

feedback_option = "thumbs"
option = st.selectbox("Choose Input Type", ["Hub URL", "Prompt Template"])


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
        placeholder="# Suppose this is a AI fashion stylist\n\nList 5 items I can pair together for an outfit for {event}. Return a list of items, separated by commas.",
    )
    if string1.strip():
        st.session_state.versions = ["", string1]
        st.session_state.source = "prompt_template"
else:
    string1 = st.text_input(
        "LangSmith Hub Repo", "", placeholder="homanp/superagent"
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

            with st.spinner("Generating evaluation suggestions..."):
                with st.expander(
                    "Generating evaluation suggestions", expanded=True
                ):
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
                        eval_functions, message_history = asyncio.run(
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

                        # Save results to session state
                        st.session_state.eval_functions += eval_functions
                        st.session_state.message_history += message_history

                        # Increment index
                        idx += 1

                st.session_state.run_ids = [run.id for run in cb.traced_runs]

if st.session_state.get("run_ids"):
    # Show eval functions
    st.write("Evaluation functions:")
    for i, function in enumerate(st.session_state.eval_functions):
        # TODO: Handle copy events
        code_id = (
            "code_"
            + hashlib.md5(str(function["code"]).encode()).hexdigest()[:6]
        )
        code_html = f"""
<pre><code id="{code_id}" class="language-python"># Needs LLM: {function['needs_llm']}
{function['code']}
</code></pre>
"""
        st.markdown(code_html, unsafe_allow_html=True)

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

    # Show CTA
    with st.form("email_for_study_form"):
        st.write(
            "This is an experimental version of the tool, built in collaboration with UC Berkeley. If you'd like to participate in an interactive prompt engineering study so we can improve the tool, please enter your email below. We will not use your email for any other purpose. For more questions, please contact Shreya Shankar at shreyashankar@berkeley.edu."
        )
        email_address = st.text_input("Email", key="email_address")
        email_submitted = st.form_submit_button("Submit")

        if email_submitted:
            if email_address:
                # Log the email address
                email_type_str = "email"
                for run_id in run_ids:
                    client.create_feedback(
                        run_id,
                        email_type_str,
                        comment=email_address,
                    )
                st.success("Thanks for your interest! We'll be in touch.")
            else:
                st.warning("Please enter an email address.")
