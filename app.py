import streamlit as st
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForTokenClassification
import torch


# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    # TODO
    processor = WhisperProcessor.from_pretrained("whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("whisper-tiny")
    model.config.forced_decoder_ids = None
    
    return processor, model


# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    # TODO
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    
    return nlp


# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
    Returns:
        str: Transcribed text from the audio file.
    """
    # TODO
    audio = uploaded_file.read()
    processor, model = load_whisper_model()
    input_values = processor(audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription


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
    #TODO
    entities = ner_pipeline(text)
    grouped_entities = {}
    for entity in entities:
        label = entity["entity"]
        text = entity["word"]
        if label not in grouped_entities:
            grouped_entities[label] = ["ORGs", "LOCs", "PERs"]            
        grouped_entities[label].append(text)
    
    return grouped_entities

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # You must replace below
    STUDENT_NAME = "Your Name"
    STUDENT_ID = "Your Student ID"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    # TODO
    # Fill here to create the streamlit application by using the functions you filled
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("Transcribe"):
            transcription = transcribe_audio(uploaded_file)
            st.write(transcription)
            nlp = load_ner_model()
            entities = extract_entities(transcription, nlp)
            st.write(entities)
            
    st.write("This is a Streamlit app for audio transcription and entity extraction.")
    st.write("Please upload an audio file to transcribe and extract entities.")
    st.write("The audio transcription is done using the Whisper model.")
    st.write("The entity extraction is done using the Named Entity Recognition (NER) model.")
    st.write("The NER model extracts entities such as ORGs, LOCs, and PERs from the transcribed text.")
    st.write("The Whisper model is a lightweight model for audio transcription.")
    st.write("The NER model is based on the BERT model for token classification.")
    st.write("The models are loaded using the Hugging Face Transformers library.")
    st.write("The app is created by Your Name 150230910 .")
    st.write("Thank you for using the app!")
    
    # TODO
    # You can add more content here if you want
    # You can also add images, videos, etc. to make your app more interactive
    


if __name__ == "__main__":
    main()
