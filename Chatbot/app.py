import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.title("KarimGBT")

generator = pipeline("text2text-generation", model="t5-small")

download_path = r'F:\Instant-DA\Projects\LLM-Chatbot'

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill", cache_dir=download_path)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill", cache_dir=download_path)

text_prompt = st.text_input("User:")
if text_prompt:
    inputs = tokenizer(text_prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"], max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("KarimGBT:", generated_text)