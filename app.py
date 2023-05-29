# ------------------------------IMPORTING-REQUIRED-LIBRARIES---------------------
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message

from pathlib import Path
from PIL import Image
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
import random

# Import OpenCV
import cv2

# LLM Model required libraries
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import (
    SimpleSequentialChain,
    SequentialChain,
)  # for allowing multiple sets of output
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
# using OpenAI
# from llm_models import openai_llm

# using huggingface
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.9})
# -----------------------------------ML-Models----------------------------------

# -----------------------------------Computer-Vision----------------------------
# Set the directory path
my_path = "."
banner_path = my_path + "/images/banner.png"

IMAGE_SHAPE = (224, 224)

BATCH_SIZE = 64  # @param {type:"integer"}

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def classify_disease_uploaded_file(upload_image):
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    def load_image(opencv_image):
        img = cv2.resize(opencv_image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        img = img / 255

        return img

    def predict(image):
        probabilities = model.predict(np.asarray([img]))[0]
        class_idx = np.argmax(probabilities)

        return {classes[class_idx]: probabilities[class_idx]}

    img = load_image(opencv_image)
    prediction = predict(img)
    disease_found = list(prediction.keys())[0].replace("_", " ")

    return (
        "PREDICTED Class: %s, Confidence: %f"
        % (
            list(prediction.keys())[0].replace("_", " "),
            list(prediction.values())[0],
        ),
        disease_found,
        img,
    )


# Loading Saved Model
saved_model_path = os.getcwd() + '/model'
model = tf.keras.models.load_model(saved_model_path)

# --------------------------------------LLM-Model-------------------------------------

# llm = openai_llm

# Create and load PDF Loader
loader = PyPDFLoader('documentation.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, collection_name='project_info')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="project_info",
    description="contains information about the whole project",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

def image_upload_response(disease_found):
    cure_template = PromptTemplate(
        input_variables=["disease_found", "wikipedia_research"],
        template="tell me about disease {disease_found} in breif, and provide a step-by-step possible treatment for it. while leveraging this wikipedia reserch:{wikipedia_research}",
    )
    # Memory
    cure_memory = ConversationBufferMemory(
        input_key="disease_found", memory_key="chat_history"
    )
    # Sequential Chain
    cure_chain = LLMChain(
        llm=llm,
        prompt=cure_template,
        verbose=True,
        output_key="title",
        memory=cure_memory,
    )

    wiki = WikipediaAPIWrapper()

    wiki_research = wiki.run(disease_found)

    return cure_chain.run(disease_found=disease_found, wikipedia_research=wiki_research)


# -------------------------------------STREAMLIT-APP-----------------------------------

st.set_page_config(page_title="Plant Care Assistant")
# Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)


activities = ["Plant Disease Assistant", "About"]
choice = st.sidebar.selectbox("Select Activty",activities)

    
if choice =='About':
    st.title("Ask about the project here. 🧐")
    prompt = st.text_input(label='Input your prompt here', value="Introduction for Plant disease detection system" )

    # If the user hits enter
    if prompt:
        # Then pass the prompt to the LLM
        response = agent_executor.run(prompt)
        # ...and write it out to the screen
        st.write(response)
        st.balloons()

        # With a streamlit expander  
        with st.expander('Information from the report'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt) 
            # Write out the first 
            st.write(search[0][0].page_content) 
if choice == 'Plant Disease Assistant':

    APP_ICON_URL = "https://i.pinimg.com/736x/4e/af/08/4eaf081c599286fd9ca84c1757c07152.jpg"
    st.write(
        "<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>",
        unsafe_allow_html=True,
    )
    st.image(APP_ICON_URL, width=80)
    # Sidebar for App Overview
    with st.sidebar:
        st.title("Plant Disease Detection App with Assistant🤖")
        st.markdown(
            """
        ## About
        This app is an LLM-powered Assistant built using:
        - [Streamlit](https://streamlit.io/)
        - [OpenAI](https://openai.com/)
        - And Much More to be added
        """
        )
        add_vertical_space(5)
        st.write(
            "Made with ❤️ by [Yash Raj](https://in.linkedin.com/in/yash-raj-2841641b3)"
        )

    # Set App title
    st.title("Dr. Greenry House🥸")

    # For uploading Image
    st.write("**Upload your Image**")
    upload_image = st.file_uploader(
        "Upload image of plant in JPG or PNG format", type=["jpg", "png"]
    )

    # Generate empty lists for generated and past.
    ## generated stores AI generated responses
    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            "Hello, I am Dr. Greenry, Please show me the image of the infected plant!"
        ]
    # past stores User's questions
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hello"]

    # Layout of input/response containers
    input_container = st.container()
    colored_header(label="", description="", color_name="blue-30")
    response_container = st.container()


    # User input
    ## Function for taking user provided prompt as input
    def get_text():
        input_text = st.text_input("You: ", "", key="input")
        return input_text


    ## Applying the user input box
    with input_container:
        user_input = get_text()


    # Response output
    ## Function for taking user prompt as input followed by producing AI generated responses
    def user_input_response(prompt):
        response = llm(prompt)
        return response


    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if st.session_state["generated"]:
            if upload_image is not None:
                # Performing Inference on the Image
                result_print, disease_found, img = classify_disease_uploaded_file(
                    upload_image
                )
                st.markdown(result_print, unsafe_allow_html=True)
                st.image(img, channels="BGR")
                output = image_upload_response(disease_found)
                image_user_input = "Can you please provide me with the cure for the {0} disease".format(disease_found.replace('_', ' '))
                st.session_state.past.append(image_user_input)
                st.session_state.generated.append(output)
                # message(st.session_state["generated"][-1], key=str(-1))
                # del upload_image
            elif user_input:
                response = user_input_response(user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
            del upload_image
