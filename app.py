import streamlit as st
from transformers import pipeline
import torch
import os
import tempfile

# ------------------------------
# Load Whisper Model
# ------------------------------
@st.cache_resource
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    try:
        whisper_pipeline = pipeline("automatic-speech-recognition", 
                                  model="openai/whisper-tiny",
                                  device="cuda" if torch.cuda.is_available() else "cpu",
                                  return_timestamps=True)  # Enable timestamp support
        return whisper_pipeline
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

# ------------------------------
# Load NER Model
# ------------------------------
@st.cache_resource
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    try:
        ner_pipeline = pipeline("ner", 
                              model="dslim/bert-base-NER",
                              device="cuda" if torch.cuda.is_available() else "cpu", 
                              aggregation_strategy="simple")  # Use simple aggregation strategy
        print("NER model loaded successfully.")
        return ner_pipeline
    except Exception as e:
        st.error(f"Error loading NER model: {str(e)}")
        return None

# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file, whisper_pipeline):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
        whisper_pipeline: Loaded whisper pipeline
    Returns:
        str: Transcribed text from the audio file.
    """
    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Transcribe the audio file with timestamps
        result = whisper_pipeline(tmp_file_path)
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        # Extract text from chunks
        if isinstance(result, dict) and 'chunks' in result:
            # Combine text from all chunks
            full_text = ' '.join(chunk['text'] for chunk in result['chunks'])
            return full_text
        elif isinstance(result, dict) and 'text' in result:
            return result['text']
        else:
            # If different format, try to get text directly
            return result['text'] if isinstance(result, dict) else str(result)
            
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    try:
        # Get entities from the text
        entities = ner_pipeline(text)

        # Initialize dictionaries for each entity type
        grouped_entities = {
            "Organizations": set(),
            "Locations": set(),
            "Persons": set()
        }

        # Group entities by type
        for entity in entities:
            # Use entity_group when aggregation_strategy="simple"
            entity_type = entity['entity_group']
            entity_text = entity['word']

            if entity_type == 'ORG':
                grouped_entities["Organizations"].add(entity_text)
            elif entity_type == 'LOC':
                grouped_entities["Locations"].add(entity_text)
            elif entity_type == 'PER':
                grouped_entities["Persons"].add(entity_text)

        return grouped_entities
    
    except Exception as e:
        st.error(f"Error during entity extraction: {str(e)}")
        return None

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # Replace with your information
    STUDENT_NAME = "Azizagha Majidov"
    STUDENT_ID = "150230910"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    # Load models
    whisper_model = load_whisper_model()
    ner_model = load_ner_model()

    if whisper_model is None or ner_model is None:
        st.error("Failed to load required models. Please refresh the page.")
        return

    # File uploader
    # File uploader with new description
    st.write("Upload a business meeting audio file to:\n")
    st.write("1. Transcribe the meeting audio into text.\n",
             "2. Extract entities (Organizations, Locations, Persons from the transcribed text.\n")
    uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=['wav'])
    if uploaded_file is not None:
        # Add a process button
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing the audio file... This may take a minute."):
                # Transcribe audio
                transcription = transcribe_audio(uploaded_file, whisper_model)
                
                if transcription:
                    # Display transcription
                    st.subheader("Transcription:")
                    st.write(transcription)
                    
                    # Extract and display entities
                    st.spinner("Extracting entities...")
                    entities = extract_entities(transcription, ner_model)
                    
                    if entities:
                        st.subheader("Extracted Entities:")
                        
                        # Display each entity type
                        for entity_type, entity_set in entities.items():
                            if entity_set:
                                st.write(f"**{entity_type}:**")
                                for entity in sorted(entity_set):
                                    st.write(f"- {entity}")
                            else:
                                st.write(f"**{entity_type}:** None found")

if __name__ == "__main__":
    main()