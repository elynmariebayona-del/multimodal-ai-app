import requests
import streamlit as st
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

TEXT_MODEL = "mistralai/mistral-7b-instruct:free"

HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"

APP_URL = "https://zmttk63suq.streamlit.app"


# ── Client ────────────────────────────────────────────────────────────────────

def get_openrouter_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def chat_with_reasoning(messages: list) -> tuple:
    try:
        clean_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
        resp = get_openrouter_client().chat.completions.create(
            model=TEXT_MODEL,
            messages=clean_messages,
            extra_headers={
                "HTTP-Referer": APP_URL,
                "X-Title": "AI Assistant",
            },
        )
        assistant_msg = resp.choices[0].message
        updated = messages + [
            {
                "role": "assistant",
                "content": assistant_msg.content,
            }
        ]
        return assistant_msg.content, updated
    except Exception as e:
        st.error(f"Chat error: {e}")
        return None, messages


def generate_image(prompt: str) -> bytes | None:
    hf_key = st.secrets.get("HUGGINGFACE_API_KEY", "")
    if not hf_key:
        st.error("HUGGINGFACE_API_KEY not found in secrets.")
        return None

    headers = {
        "Authorization": f"Bearer {hf_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=120,
        )
        if response.status_code == 503:
            st.warning("Model is loading on Hugging Face (~20s), please try again.")
            return None
        response.raise_for_status()
        return response.content

    except requests.exceptions.ConnectionError as e:
        st.error(f"Cannot reach Hugging Face API: {e}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again.")
        return None
    except requests.HTTPError as e:
        st.error(f"Hugging Face API error {response.status_code}: {e}")
        return None


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")

tab_chat, tab_image = st.tabs(["💬 Chat", "🎨 Image Generation"])


# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

with tab_chat:
    st.subheader("Chat with Reasoning")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] in ("user", "assistant"):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, updated = chat_with_reasoning(st.session_state.messages)
            if reply:
                st.markdown(reply)
                st.session_state.messages = updated

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Tab 2: Image Generation ───────────────────────────────────────────────────

with tab_image:
    st.subheader("Image Generation")

    img_prompt = st.text_area("Describe the image you want", height=100)

    if st.button("✨ Generate Image", disabled=not img_prompt.strip()):
        with st.spinner("Generating image... (may take 20–60s on first run)"):
            img_bytes = generate_image(img_prompt)
            if img_bytes:
                st.image(img_bytes, caption=img_prompt, use_column_width=True)
                st.download_button(
                    label="⬇️ Download Image",
                    data=img_bytes,
                    file_name="generated.png",
                    mime="image/png",
                )
