import io
import base64
import requests
import streamlit as st
from openai import OpenAI

# ── Clients ───────────────────────────────────────────────────────────────────

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
)

HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
TEXT_MODEL = "openai/gpt-oss-120b:free"


# ── Helpers ───────────────────────────────────────────────────────────────────

def chat_with_reasoning(messages: list) -> tuple[str, list]:
    """Send messages with reasoning enabled. Returns (reply_text, updated_messages)."""
    resp = openrouter_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        extra_body={"reasoning": {"enabled": True}},
    )
    assistant_msg = resp.choices[0].message

    updated = messages + [
        {
            "role": "assistant",
            "content": assistant_msg.content,
            "reasoning_details": assistant_msg.reasoning_details,
        }
    ]
    return assistant_msg.content, updated


def generate_image(prompt: str) -> bytes:
    """Generate an image via Hugging Face and return raw bytes."""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": prompt},
        timeout=120,
    )
    response.raise_for_status()
    return response.content


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")

tab_chat, tab_image = st.tabs(["💬 Chat", "🎨 Image Generation"])


# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

with tab_chat:
    st.subheader("Chat with Reasoning")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history (skip reasoning_details in display)
    for msg in st.session_state.messages:
        if msg["role"] in ("user", "assistant"):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, updated_messages = chat_with_reasoning(st.session_state.messages)
            st.markdown(reply)

        st.session_state.messages = updated_messages

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Tab 2: Image Generation ───────────────────────────────────────────────────

with tab_image:
    st.subheader("Image Generation")

    img_prompt = st.text_area("Describe the image you want", height=100)

    if st.button("✨ Generate Image", disabled=not img_prompt.strip()):
        with st.spinner("Generating image..."):
            try:
                img_bytes = generate_image(img_prompt)
                st.image(img_bytes, caption=img_prompt, use_column_width=True)

                st.download_button(
                    label="⬇️ Download Image",
                    data=img_bytes,
                    file_name="generated.png",
                    mime="image/png",
                )
            except requests.HTTPError as e:
                st.error(f"Image generation failed: {e}")
