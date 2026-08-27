"""
SparkAI-47M-Llama-Instruct — Gradio Chat UI

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
REPO_ID = "vedantjadhav701/SparkAI-47m-llama-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ---- Load model + tokenizer once at startup ----
print(f"Loading {REPO_ID} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(REPO_ID, torch_dtype=DTYPE).to(DEVICE)
model.eval()
print("Model loaded.")


def generate_response(history, max_new_tokens, temperature, top_p, repetition_penalty):
    """
    Streams tokens back to the Gradio chat UI using chat template formatting.
    """
    messages = []
    # Build OpenAI format messages list from history (excluding last empty assistant turn if present)
    turns = history[:-1] if (history and (
        (isinstance(history[-1], dict) and history[-1].get("content") == "") or
        (hasattr(history[-1], "content") and getattr(history[-1], "content") == "")
    )) else history

    for turn in turns:
        if isinstance(turn, dict):
            messages.append({"role": turn.get("role", "user"), "content": str(turn.get("content", ""))})
        elif hasattr(turn, "role") and hasattr(turn, "content"):
            messages.append({"role": getattr(turn, "role", "user"), "content": str(getattr(turn, "content", ""))})
        elif isinstance(turn, (list, tuple)):
            if len(turn) > 0 and turn[0]:
                messages.append({"role": "user", "content": str(turn[0])})
            if len(turn) > 1 and turn[1]:
                messages.append({"role": "assistant", "content": str(turn[1])})

    if not messages:
        return

    # Apply chat template if available
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to last user message content if chat_template fails
        prompt = messages[-1]["content"]

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

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


with gr.Blocks(title="SparkAI-47M-Llama-Instruct Chat") as demo:
    gr.Markdown(
        f"""
        # SparkAI-47M-Llama-Instruct
        Instruction-tuned checkpoint of **SparkAI-47M-Llama** (~48M parameters).
        Fine-tuned for chat and instruction following.

        Model: [`{REPO_ID}`](https://huggingface.co/{REPO_ID})
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, elem_id="chat-history")
            msg = gr.Textbox(
                label="Prompt",
                placeholder="Ask something, e.g. 'What is machine learning?'",
                lines=2,
                elem_id="prompt-input",
            )
            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary", elem_id="submit-btn")
                clear_btn = gr.Button("Clear", elem_id="clear-btn")

        with gr.Column(scale=1):
            gr.Markdown("### Generation settings")
            max_new_tokens = gr.Slider(
                minimum=10, maximum=300, value=120, step=10, label="Max new tokens", elem_id="slider-max-tokens"
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.7, step=0.05, label="Temperature", elem_id="slider-temperature"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p", elem_id="slider-top-p"
            )
            repetition_penalty = gr.Slider(
                minimum=1.0, maximum=2.0, value=1.15, step=0.05, label="Repetition penalty", elem_id="slider-rep-penalty"
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

        history.append({"role": "assistant", "content": ""})
        for partial in generate_response(
            history, max_new_tokens, temperature, top_p, repetition_penalty
        ):
            if isinstance(history[-1], dict):
                history[-1]["content"] = partial
            elif hasattr(history[-1], "content"):
                setattr(history[-1], "content", partial)
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