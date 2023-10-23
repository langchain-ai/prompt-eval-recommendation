from typing import Optional
import streamlit as st
from langchain.callbacks.manager import collect_runs
from langchain.prompts import BasePromptTemplate
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from eval_suggestions_azure import suggest_evals
from langchain import hub
from langchainhub import Client as HubClient
import asyncio
import functools

client = Client()
hub_client = HubClient()

st.set_page_config(
    page_title="Capturing User Feedback",
    page_icon="🦜️️🛠️",
)

st.subheader("🦜🛠️ Get Suggested Evalution Functions")
st.write(
    "This tool will suggest binary eval functions for your prompt, that you can run on all future responses to your prompt. It works best when given the version history of your prompt (i.e., through a LangSmith Hub repo), but you can also use it with a single prompt template."
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

feedback_option = "faces"
option = st.selectbox("Choose Input Type", ["Hub URL", "Prompt Template"])
versions = []


def list_versions(repo_full_name: str):
    hashes = [
        v["commit_hash"]
        for v in hub_client.list_commits(repo_full_name)["commits"]
    ]
    return [hash[:7] for hash in hashes]


feedback_key = f"feedback_{st.session_state.get('run_id')}"


def on_feedback_submit(feedback, run_id):
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


if feedback_key in st.session_state:
    on_feedback_submit(st.session_state[feedback_key], st.session_state.run_id)
if option == "Prompt Template":
    string1 = st.text_area(
        "Prompt Template",
        "",
        placeholder="# Suppose this is a AI fashion stylist\n\nList 5 items I can pair together for an outfit for {event}. Return a list of items, separated by commas.",
    )
    if string1.strip():
        versions = ["", string1]
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
            with st.expander("Versions"):
                st.write(hashes)
            try:
                prompts = []
                for hash_ in hashes:
                    prompts.append(hub.pull(string1 + ":" + hash_))
            except Exception as e:
                st.error(repr(e))
                exit()
            # TODO: Actually incorporate in predictions/recommendations
            # We are just using blank to latest here
            prompt: BasePromptTemplate = prompts[-1]
            versions = [
                "",
                prompt.format(
                    **{k: f"{{{k}}}" for k in prompt.input_variables}
                ),
            ]

if versions:
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
            with st.expander("Processing prompt", expanded=True):
                char_response = []
                message_placeholder = st.empty()
                function_response = []
                eval_placeholder = st.empty()
                eval_functions, message_history = asyncio.run(
                    # TODO: handle multiple versions
                    suggest_evals(
                        versions[0],
                        versions[1],
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
        st.write("Evaluation functions:")
        for i, function in enumerate(eval_functions):
            # TODO: Handle copy events
            st.markdown(
                f"```python\n# Needs LLM: {function['needs_llm']}\n{function['code']}\n```"
            )
        st.session_state.run_id = cb.traced_runs[0].id

    if st.session_state.get("run_id"):
        # Show full message history
        with st.expander("Full message history", expanded=False):
            # Get the "assistant" messages
            st.write(
                "Here are the full GPT-4 responses to (1) categorize the prompt additions and (2) generate evaluation functions."
            )
            st.write([m for m in message_history if m["role"] == "assistant"])

        run_id = st.session_state.run_id

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id

    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{run_id}",
        on_submit=on_feedback_submit,
    )
    print("Feedback called")

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
                email_feedback_record = client.create_feedback(
                    run_id,
                    email_type_str,
                    comment=email_address,
                )
                st.session_state.email_feedback = {
                    "feedback_id": str(email_feedback_record.id),
                    "email_address": email_address,
                }

                st.success("Thanks for your interest! We'll be in touch.")
            else:
                st.warning("Please enter an email address.")
