import streamlit as st
from openai import OpenAI
from huggingface_hub import InferenceClient

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Multi-Modal AI App", page_icon="🤖", layout="wide")

# --- SECURE API KEY HANDLING ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
HF_TOKEN = st.secrets.get("HF_TOKEN")

if not OPENROUTER_API_KEY or not HF_TOKEN:
    st.error("⚠️ API Keys missing! Please add OPENROUTER_API_KEY and HF_TOKEN to your `.streamlit/secrets.toml` file.")
    st.stop()

# Initialize the OpenRouter client for Chat
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Initialize the Hugging Face client for Image Generation
hf_client = InferenceClient(token=HF_TOKEN)

# --- UI HEADER ---
st.title("🤖 NLP: Multi-Modal AI App")
st.caption("Artificial Intelligence 10.0 - Chat & Image Generator")

# --- SIDEBAR MODE SELECTION ---
mode = st.sidebar.radio("Choose Mode:", ["💬 Chat with LLM", "🎨 Image Generator"])

# ==========================================
# MODE 1: CHAT (OpenRouter - baidu/cobuddy:free)
# ==========================================
if mode == "💬 Chat with LLM":
    st.header("Chat with the Large Language Model")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat messages from history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is on your mind?"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Prepare messages for the API
        api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]

        # Get LLM response
        with st.spinner("Thinking..."):
            try:
                response = openrouter_client.chat.completions.create(
                    model="openai/gpt-oss-120b:free",
                    messages=api_messages,
                    extra_body={"reasoning": {"enabled": True}}
                )
                assistant_reply = response.choices[0].message.content
            except Exception as e:
                assistant_reply = f"Sorry, an error occurred: {e}"

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        
        # Add to history
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})


# ==========================================
# MODE 2: IMAGE GENERATION (Hugging Face)
# ==========================================
elif mode == "🎨 Image Generator":
    st.header("Generate Images from Text")
    st.info("Using `stabilityai/stable-diffusion-xl-base-1.0` via Hugging Face Free Inference API.")
    
    img_prompt = st.text_input("Describe the image you want to generate:", key="image_prompt")
    
    if st.button("Generate Image", type="primary"):
        if img_prompt:
            with st.spinner("Creating your image... (May take 10-30 seconds if the model is waking up)"):
                try:
                    # Call Hugging Face text_to_image API
                    image = hf_client.text_to_image(
                        prompt=img_prompt, 
                        model="stabilityai/stable-diffusion-xl-base-1.0"
                    )
                    
                    # Display the PIL Image object directly in Streamlit
                    st.image(image, caption=img_prompt, use_column_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred while generating the image: {e}")
                    st.warning("💡 *Tip: Hugging Face free tier models sometimes need to 'warm up'. Try clicking Generate again in 30 seconds.*")
        else:
            st.warning("Please enter a prompt first!")
