from typing import Optional

import streamlit as st
# from streamlit_drawable_canvas import st_canvas
from PIL import Image
from utils import generate

DEFAULT_PROMPT = ""
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
OUTPUT_IMAGE_KEY = "output_img"
LOADED_IMAGE_KEY = "loaded_image"

## test url : https://www.youtube.com/watch?v=ayKJsoa-GU0, https://www.youtube.com/watch?v=BG7273yDpdA

def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img


def prompt_and_generate_button(
                        prompt=None,
                        text_label = "Enter Link here",
                        pipeline_name=None,
                        command=None,
                        image_input=None,
                        mask_input=None,
                        negative_prompt=None,
                        steps=50,
                        width=768,
                        height=768,
                        guidance_scale=7.5,
                        enable_attention_slicing=False,
                        enable_cpu_offload=False,
                        version="2.1",
                        strength=1.0,
                        prefix=None,
                        ):
    prompt = st.text_area(
        text_label,
        value=DEFAULT_PROMPT,
        key=f"{prefix}-prompt",
    )

    if st.button("Generate ", key=f"{prefix}-btn"):
        # st.video(prompt)
        with st.spinner("Generating ..."):
            # common function for generation
            output = generate(
                prompt=prompt,
                pipeline_name=pipeline_name,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                enable_attention_slicing=enable_attention_slicing,
                enable_cpu_offload=enable_cpu_offload,
                command=command,
                image_input=image_input,
                mask_input=mask_input,
                width=width,
                height=height,
                version=version,
                strength=strength,
                prefix=prefix,
            )
        return output



def image_uploader(prefix):
    image = st.file_uploader("Image", ["jpg", "png"], key=f"{prefix}-uploader")
    if image:
        image = Image.open(image)
        print(f"loaded input image of size ({image.width}, {image.height})")
        return image

    return get_image(LOADED_IMAGE_KEY)




def transcript_tab():
    prefix = "transcript"    
    # function for generating transcript using llm
    prompt_and_generate_button(
        prefix=prefix,
        version='deepgram_whisper'
    )


def scenic_question_answer_tab():
    prefix = "scenic-question-answer"
    # implement here
    prompt_and_generate_button(
        prefix=prefix
    )
        



def caption_thumbnail_tab():
    prefix = "caption-thunmnail"
#      start work here
    prompt_and_generate_button(
        prefix=prefix,
        version='caption_generator'
    )

def eval_metrics_tab():
    prefix = "eval-metric"
    # use bert rouge and other metrics
    prompt_and_generate_button(
        prefix=prefix,
        version='eval'
    )
    
def chatbot_tab():
    prefix = "chatbot"
    #implement chatbot with llama 3.1
    prompt_and_generate_button(
        prefix=prefix,
        version='chatbot',
        text_label='Ask here'
    )
    


def main():
    st.set_page_config(layout="wide")
    st.title("New-Tube")

    tab1, tab3, tab4, tab5 = st.tabs(
        ["Transcript", "Caption & Thumbnail","Eval-metrics","Chat-Bot"]
    )
    with tab1:
        transcript_tab()

    with tab3:
        caption_thumbnail_tab()
    
    with tab4:
        eval_metrics_tab()
    
    with tab5:
        chatbot_tab()
        

    with st.sidebar:
        st.header("Output")
        # do something here

if __name__ == "__main__":
    main()