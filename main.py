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

client = Client()
hub_client = HubClient()

st.set_page_config(
    page_title="Capturing User Feedback",
    page_icon="🦜️️🛠️",
)

st.subheader("🦜🛠️ Get Suggested Evalution Functions")

feedback_option = "faces"
option = st.selectbox("Choose Input Type", ["Hub URL", "Prompt Template"])
versions = []


def list_versions(repo_full_name: str):
    hashes = [
        v["commit_hash"] for v in hub_client.list_commits(repo_full_name)["commits"]
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
    string1 = st.text_area("Template", "", placeholder="Hello, {input}!")
    if string1.strip():
        versions = ["", string1]
else:
    string1 = st.text_input("LangSmith Hub Repo", "", placeholder="homanp/superagent")
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
                    prompts.append(hub.pull(string1+":"+hash_))
            except Exception as e:
                st.error(repr(e))
                exit()
            # TODO: Actually incorporate in predictions/recommendations
            # We are just using blank to latest here
            prompt: BasePromptTemplate = prompts[-1]
            versions = [
                "",
                prompt.format(**{k: f"{{{k}}}" for k in prompt.input_variables}),
            ]

if versions:
    with collect_runs() as cb:

        def typing_callback(text: Optional[str], agg: list, message_placeholder):
            if text is None:
                # Erase
                message_placeholder.markdown("")
                return
            agg.append(text)
            message_placeholder.markdown("".join(agg) + "▌")

        with st.spinner("Generating evaluation suggestions..."):
            with st.expander("Processing prompt"):
                char_response = []
                message_placeholder = st.empty()
                function_response = []
                eval_placeholder = st.empty()
                eval_functions, _ = asyncio.run(
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
        for function in eval_functions:
            # TODO: Handle copy events
            st.markdown(
                f"```python\n# Needs LLM: {function['needs_llm']}\n{function['code']}\n```"
            )
        st.session_state.run_id = cb.traced_runs[0].id

    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id
        
        feedback = streamlit_feedback(
            feedback_type=feedback_option,
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{run_id}",
            on_submit=on_feedback_submit,
        )
        print("Feedback called", feedback)
        
