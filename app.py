"""
SparkAI-47M-Llama — Gradio Chat UI

Run:
    pip install gradio transformers torch accelerate
    python app.py

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# ---- Config ----
REPO_ID = "vedantjadhav701/SparkAI-47m-llama-10b-token"  # change to whichever checkpoint you want to serve
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ---- Load model + tokenizer once at startup ----
print(f"Loading {REPO_ID} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(REPO_ID, dtype=DTYPE).to(DEVICE)
model.eval()
print("Model loaded.")


def generate_response(message, history, max_new_tokens, temperature, top_p, repetition_penalty):
    """
    Streams tokens back to the Gradio chat UI as they're generated.
    Note: this is a base (non-instruction-tuned) model, so it completes
    text rather than following chat-formatted instructions — treat the
    input as a prompt continuation, not a strict Q&A exchange.
    """
    if not isinstance(message, str):
        if isinstance(message, dict) and "content" in message:
            message = str(message["content"])
        elif isinstance(message, (list, tuple)) and len(message) > 0:
            message = str(message[0])
        else:
            message = str(message)

    inputs = tokenizer(message, return_tensors="pt").to(DEVICE)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text


with gr.Blocks(title="SparkAI-47M-Llama Chat") as demo:
    gr.Markdown(
        f"""
        # SparkAI-47M-Llama
        A ~47M parameter LLaMA-style model trained from scratch on ~10B tokens
        (FineWeb-Edu + Cosmopedia-v2). **Base model, not instruction-tuned** —
        expect text continuation rather than assistant-style chat behavior.

        Model: [`{REPO_ID}`](https://huggingface.co/{REPO_ID})
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, type="messages", elem_id="chat-history")
            msg = gr.Textbox(
                label="Prompt",
                placeholder="Type a sentence to continue, e.g. 'The capital of France is'",
                lines=2,
                elem_id="prompt-input",
            )
            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary", elem_id="submit-btn")
                clear_btn = gr.Button("Clear", elem_id="clear-btn")

        with gr.Column(scale=1):
            gr.Markdown("### Generation settings")
            max_new_tokens = gr.Slider(
                minimum=10, maximum=200, value=60, step=10, label="Max new tokens", elem_id="slider-max-tokens"
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.8, step=0.05, label="Temperature", elem_id="slider-temperature"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p", elem_id="slider-top-p"
            )
            repetition_penalty = gr.Slider(
                minimum=1.0, maximum=2.0, value=1.2, step=0.05, label="Repetition penalty", elem_id="slider-rep-penalty"
            )

    def user_submit(message, history):
        if not message or not str(message).strip():
            return "", history
        history = history or []
        history.append({"role": "user", "content": str(message)})
        return "", history

    def bot_respond(history, max_new_tokens, temperature, top_p, repetition_penalty):
        if not history:
            return
        last_turn = history[-1]
        if isinstance(last_turn, dict):
            user_message = last_turn.get("content", "")
        elif isinstance(last_turn, (list, tuple)) and len(last_turn) > 0:
            user_message = last_turn[0]
        else:
            user_message = str(last_turn)

        user_message = str(user_message) if user_message is not None else ""

        history.append({"role": "assistant", "content": ""})
        for partial in generate_response(
            user_message, history, max_new_tokens, temperature, top_p, repetition_penalty
        ):
            history[-1]["content"] = partial
            yield history

    msg.submit(
        user_submit, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        bot_respond,
        [chatbot, max_new_tokens, temperature, top_p, repetition_penalty],
        chatbot,
    )

    submit_btn.click(
        user_submit, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        bot_respond,
        [chatbot, max_new_tokens, temperature, top_p, repetition_penalty],
        chatbot,
    )

    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=False, theme=gr.themes.Soft())