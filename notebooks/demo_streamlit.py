import torch
import threading
import streamlit as st
import pandas as pd
import os

from PIL import Image
from io import StringIO
from threading import Thread
from transformers import AutoProcessor, AutoModelForCausalLM


# LOAD ALL MODELS AND PROCESSORS

## ********* FLORENCE-2 *********
def load_florence_2():
    model_id = 'microsoft/Florence-2-base-ft'
    FLORENCE_2 = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
    FLORENCE_2_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return FLORENCE_2, FLORENCE_2_PROCESSOR
    

def florence_caption(image, task_prompt="<MORE_DETAILED_CAPTION>", text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = FLORENCE_2_PROCESSOR(text=prompt, images=image, return_tensors="pt").to("cuda:0")

    generated_ids = FLORENCE_2.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1500,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = FLORENCE_2_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = FLORENCE_2_PROCESSOR.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer["<MORE_DETAILED_CAPTION>"]

FLORENCE_2, FLORENCE_2_PROCESSOR = load_florence_2()

## ********* GIT BASE *********
def load_git_base():
    model_id = 'microsoft/git-base-coco'
    GIT_BASE = AutoModelForCausalLM.from_pretrained(model_id)
    GIT_BASE_PROCESSOR = AutoProcessor.from_pretrained(model_id)
    return GIT_BASE, GIT_BASE_PROCESSOR

def gitbase_caption(image):
    pixel_values = GIT_BASE_PROCESSOR(images=image, return_tensors="pt").pixel_values
    generated_ids = GIT_BASE.generate(pixel_values=pixel_values, max_length=500)
    generated_caption = GIT_BASE_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption

GIT_BASE, GIT_BASE_PROCESSOR = load_git_base()

# STREAMLIT APP
st.title('Demo Streamlit')

with st.sidebar:
    st.header('Settings')

    model = st.selectbox("Select your model", ('Florence-2', 'Git Base'))
    print(f"Selected model: {model}")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", 'png', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    flag = st.button("Generate Caption")
    if flag and model == 'Florence-2':
        caption = florence_caption(image)
        st.text_area("Image caption:",  caption)

    elif flag and model == 'Git Base':
        caption = gitbase_caption(image)
        st.text_area("Image caption:", caption)