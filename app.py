import streamlit as st
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.firecrawl import FirecrawlTools
import google.generativeai as genai
import io
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys from Streamlit Secrets
FIRECRAWL_API_KEY = st.secrets["FIRCRAWL_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Configure API Key
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page Configuration
st.set_page_config(
    page_title="AI Shopping Partner",
    page_icon="🤖🛍️",
    layout="centered"
)

st.title("AI Shopping Partner")
st.header("Powered by Agno and Google Gemini")

def get_gemini_response(api_key, prompt, image_path):
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    response = model.generate_content([prompt, image_bytes])
    return response.text

def initialize_agent():
    return Agent(
        name="Shopping Partner",
        model=Gemini(id="gemini-2.0-flash-exp"),  
        instructions=[
            "You are a product recommender agent specializing in finding products that match user preferences.",
            "Prioritize finding products that satisfy as many user requirements as possible, but ensure a minimum match of 50%.",
            "Search for products only from authentic and trusted e-commerce websites such as Google Shopping, Amazon, Flipkart, Myntra, Meesho, Nike, and other reputable platforms.",
            "Verify that each product recommendation is in stock and available for purchase.",
            "Avoid suggesting counterfeit or unverified products.",
            "Clearly mention the key attributes of each product (e.g., price, brand, features) in the response.",
            "Format the recommendations neatly and ensure clarity for ease of user understanding.",
        ],
        tools=[FirecrawlTools()],
        markdown=True
    )

# Initialize the Agent
multimodal_Agent = initialize_agent()

# File Uploader
image_file = st.file_uploader("Upload an image file to analyze and provide relevant shopping links", type=["jpg", "jpeg", "png"], help="Upload max 200MB image for AI Analysis")

prompt = "What is in this photo?"
if image_file is not None:
    try:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_container_width=False, width=400)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_image_path = temp_file.name
        
        with st.spinner("AI is processing this image and gathering insights..."):
            response = get_gemini_response(API_KEY, prompt, temp_image_path)
            st.write(f"Product Identified using AI: {response}")
        
        os.remove(temp_image_path)
    except Exception as e:
        st.error(f"Error: Unable to open image. {e}")
    
    promptColor = st.text_input("What color are you looking for?", key="inputcolor")
    promptPurpose = st.text_input("For what purpose are you looking for this product?", key="inputpurpose")
    promptBudget = st.text_input("What is your budget?", key="inputbudget")
    user_query = st.text_area(
        "What specific insights are you looking for from the image?",                 
        placeholder="Ask any questions related to the image content. The AI agent will analyze and gather more context if necessary",
        help="Share the specific questions or details you want to explore from the image."
    )
    
    if st.button("Search this Product", key="analyse_image_button"):
        if not user_query:
            st.warning("Please enter a query to analyze this image")
        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    image.save(temp_file.name)
                    temp_image_path = temp_file.name
                
                with st.spinner("AI is processing this image and gathering insights..."):
                    analysis_prompt = f"""
                    I am looking for {response} with the following preferences:
                    Color: {promptColor} 
                    Purpose: {promptPurpose}
                    Budget: {promptBudget}
                    Can you provide recommendations? Always include hyperlinks to the product.
                    {user_query}
                    """
                    
                    response = multimodal_Agent.run(analysis_prompt, image=temp_image_path)
                
                st.subheader("Relevant search links for the product")
                st.markdown(response.content)
                os.remove(temp_image_path)
            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
    
    # Customize text area height
    st.markdown(
        """
        <style>
        .stTextArea textarea{
            height:100px;   
        }
        </style>
        """,
        unsafe_allow_html=True
    )
