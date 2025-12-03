import sys
import os
from pathlib import Path
import time
import re

# Add src to path to import chatterbox
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

import streamlit as st
import torch
import numpy as np
import soundfile as sf
import google.generativeai as genai

# Import Chatterbox
from chatterbox.tts import ChatterboxTTS

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="StoryGen & Chatterbox TTS", layout="wide")

@st.cache_resource
def load_tts_model():
    """Load the Chatterbox TTS model."""
    return ChatterboxTTS.from_pretrained(DEVICE)

def generate_story(prompt, max_words=2000, api_key=None):
    """Generate a story using Gemini API."""
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        return None
    
    genai.configure(api_key=api_key)
    
    # Try to find an available text generation model
    try:
        with st.spinner("Finding available Gemini models..."):
            available_models = genai.list_models()
            text_models = [m for m in available_models if 'generateContent' in m.supported_generation_methods]
            
            if not text_models:
                st.error("No text generation models available for your API key. Please check your key and billing settings.")
                return None
            
            # Use the first available model (usually best/newest)
            model_name = text_models[0].name
            st.info(f"Using model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
    except Exception as e:
        st.error(f"Error accessing Gemini API: {e}")
        st.info("Trying fallback models...")
        # Try common model names as fallback
        for fallback_name in ['models/gemini-1.5-pro-latest', 'models/gemini-1.5-flash-latest', 'models/gemini-pro']:
            try:
                model = genai.GenerativeModel(fallback_name)
                st.success(f"Using fallback: {fallback_name}")
                break
            except:
                continue
        else:
            st.error("Could not find any working Gemini model. Please check your API key.")
            return None
    
    full_prompt = f"""Write a detailed, engaging story about: {prompt}

The story should be approximately {max_words} words long. Make it creative, vivid, and suitable for narration.
Focus on storytelling with clear narrative flow, interesting characters, and descriptive language."""
    
    # Generate with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Generating story... (attempt {attempt + 1}/{max_retries})"):
                response = model.generate_content(full_prompt)
                generated_text = response.text
                return generated_text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # Progressive wait: 30s, 60s, 90s
                    st.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Rate limit exceeded. Please wait a few minutes or try a different API key.")
                    return None
            else:
                st.error(f"Error generating story: {e}")
                return None
    
    return None

def chunk_text(text, max_chars=300):
    """Split text into chunks to avoid TTS timeouts/freezes."""
    # Split by sentence endings first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def generate_audio_chunked(model, text, ref_audio_path=None):
    """Generate audio for long text by chunking."""
    chunks = chunk_text(text)
    audio_segments = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare conditionals once if reference audio is provided
    if ref_audio_path:
        model.prepare_conditionals(ref_audio_path)
    
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Synthesizing chunk {i+1}/{total_chunks}...")
        
        # Generate audio for the chunk
        # Note: We don't pass audio_prompt_path here if we already prepared conditionals
        # to avoid re-processing it every time.
        try:
            wav = model.generate(
                chunk,
                audio_prompt_path=None, # Already prepared
                temperature=0.7, # Default
            )
            # wav is (1, samples) tensor
            audio_segments.append(wav.squeeze(0).cpu().numpy())
        except Exception as e:
            st.error(f"Error generating chunk {i+1}: {e}")
            # Continue or break? Let's continue with silence or skip
            continue
            
        progress_bar.progress((i + 1) / total_chunks)
        
    if not audio_segments:
        return None, None

    # Concatenate all segments
    full_audio = np.concatenate(audio_segments)
    return model.sr, full_audio

def main():
    st.title("📚 StoryGen & Chatterbox TTS")
    st.markdown("Generate long stories and convert them to speech using Chatterbox.")

    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Gemini API Key
        st.subheader("🤖 Gemini API")
        gemini_api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Get your API key from https://aistudio.google.com/apikey"
        )
        
        story_len = st.slider("Approx Story Length (Words)", 500, 3000, 1500)
        
        st.divider()
        st.subheader("🎙️ Voice Style")
        
        # 1. Voice Style Library
        styles_dir = Path("customapp/voice_styles")
        styles_dir.mkdir(parents=True, exist_ok=True)
        
        saved_styles = [f.name for f in styles_dir.glob("*.wav")]
        selected_style_name = st.selectbox(
            "Load Saved Style", 
            ["None"] + saved_styles,
            index=0
        )
        
        # 2. Input Methods
        st.markdown("**Or create a new style:**")
        input_method = st.radio("Input Method", ["Upload WAV", "Record Voice"], horizontal=True)
        
        current_ref_path = None
        
        if input_method == "Upload WAV":
            uploaded_file = st.file_uploader("Upload Reference Audio (WAV)", type=["wav"])
            if uploaded_file:
                # Save to temp
                temp_dir = Path("temp_audio")
                temp_dir.mkdir(exist_ok=True)
                current_ref_path = temp_dir / uploaded_file.name
                with open(current_ref_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Loaded: {uploaded_file.name}")
                
        elif input_method == "Record Voice":
            # st.audio_input is available in newer Streamlit versions
            # Fallback for older versions or if not available: use file uploader or custom component
            # But assuming standard streamlit >= 1.40
            try:
                audio_value = st.audio_input("Record a sample (5-10s)")
                if audio_value:
                    temp_dir = Path("temp_audio")
                    temp_dir.mkdir(exist_ok=True)
                    current_ref_path = temp_dir / "recorded_sample.wav"
                    with open(current_ref_path, "wb") as f:
                        f.write(audio_value.getbuffer())
                    st.success("Recording loaded!")
            except AttributeError:
                st.error("Your Streamlit version might be too old for st.audio_input. Please update or use Upload.")

        # Logic to determine final reference path
        final_ref_path = None
        
        # Priority: New Input > Selected Saved Style
        if current_ref_path:
            final_ref_path = current_ref_path
        elif selected_style_name != "None":
            final_ref_path = styles_dir / selected_style_name
            st.info(f"Using saved style: {selected_style_name}")

        # 3. Save Feature
        if current_ref_path:
            st.divider()
            new_style_name = st.text_input("Name this voice style")
            if st.button("Save Voice Style"):
                if new_style_name:
                    save_path = styles_dir / f"{new_style_name}.wav"
                    # Copy temp file to styles dir
                    with open(current_ref_path, "rb") as src, open(save_path, "wb") as dst:
                        dst.write(src.read())
                    st.success(f"Saved style: {new_style_name}")
                    time.sleep(1)
                    st.rerun() # Refresh to show in dropdown
                else:
                    st.error("Please enter a name.")

    # Main Interface
    prompt = st.text_area("Enter a story prompt:", "A brave knight who is afraid of dragons...")
    
    if st.button("Generate Story & Audio"):
        if not final_ref_path:
            st.warning("Please select a voice style, upload a file, or record your voice.")
            return

        # 1. Generate Story
        story_text = generate_story(prompt, max_words=story_len, api_key=gemini_api_key)
        if not story_text:
            return
            
        st.subheader("Generated Story")
        st.write(story_text)
        
        # 2. Generate Audio
        st.subheader("Audio Generation")
        tts_model = load_tts_model()
        
        sr, audio_data = generate_audio_chunked(tts_model, story_text, str(final_ref_path))
        
        if audio_data is not None:
            st.success("Audio generation complete!")
            st.audio(audio_data, sample_rate=sr)
            
            # Option to download
            # Save to file first
            output_path = "output_story.wav"
            sf.write(output_path, audio_data, sr)
            with open(output_path, "rb") as f:
                st.download_button("Download Audio", f, file_name="story.wav")

if __name__ == "__main__":
    main()
