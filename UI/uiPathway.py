import gradio as gr
import requests
import time
import os
from loguru import logger 
from openai import OpenAI
from collections import deque
import json
from exa_py import Exa
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_GHAR"))
EXA_KEY="0329457a-1b25-49a1-9c74-348c53fcc9ba"
JINA_KEY="jina_033fa4291a624e48b715fc9024d06f7cVqSbN_qX-BiwGmvOHEcam23HCyJu"
#JINA_KEY="abc"

HOST="127.0.0.1"
PORT="8001"

# log_file = "./app/shared_logs/logs/log_text.md"
log_file = "./logs/log_text.md"

def make_visible():
    return gr.update(visible=True)

def success_message():
    return gr.update(value="Link uploaded successfully !!")

def escape_markdown(text):
    # Escape Markdown special characters
    special_characters = ['\\', '`', '_', '{', '}', '[', ']', '''"''']
    for char in special_characters:
        text = text.replace(char, f"\\{char}")
    return text

def log_read():
    with open(log_file, 'r') as file:
        # Use a deque to keep the last 100 lines
        last_lines = deque(file, maxlen=100)

    combined_lines = ''.join(last_lines)
    escaped_text = escape_markdown(combined_lines)
    markdown_text = f"\n{escaped_text}\n"

    return markdown_text

def clear_log_file(file_path: str) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create the file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass  # Create an empty file
    
    # Clear the content of the file
    with open(file_path, "r+") as f:
        f.seek(0)
        f.truncate()


class ChatBot:
    def __init__(self):
        self.current_query = ""
        self.collection_name_global=None


    def upload_fastapi(self, file_input, drive_link):
        print(file_input)
        if file_input is None:
            return "No file uploaded."
        #send file path to FastAPI
        with open(file_input, "rb") as file:
            self.collection_name_global = requests.post(
                url=f"http://{HOST}:{PORT}/upload?drive_link={drive_link}", 
                # json={
                #     "file": file,
                #     "link": drive_link
                #     })#.text.strip('''"''')
                files={"file": file}
            ).text.strip('''"''')
        logger.info(f"This ids from upload fast_api: {self.collection_name_global}")
        return gr.update(value=f"File uploaded successfully! \n" )    


    def call_primary_api(self, url, query):
        headers = {
            'Authorization': "Bearer "+ JINA_KEY
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError for bad responses
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful AI assistant. Summarize the given search results about the {query}"},
                {"role": "user", "content": f"Search results: {response.text}\n"}
            ]
        )
        return response.choices[0].message.content


    def web_search(self, query):
        try:
            logger.debug("Trying primary API...")
            url = "https://s.jina.ai/" + query
            logger.debug(f"Query: {query}")
            result = self.call_primary_api(url, query)
            logger.success("JINA working..")
        except requests.RequestException as e:
            logger.debug(f"Primary API failed: {e}")
            logger.debug("Trying backup API...")
            exa = Exa(api_key=EXA_KEY)

            result = exa.search_and_contents(
                query,
                type="neural",
                use_autoprompt=True,
                num_results=3,
                text=True
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Summarize the given search results about user_message"},
                    {"role": "user", "content": f"Search results: {result}\n"}
                ]
            )

            result = response.choices[0].message.content
            logger.debug(f"Backup API result: {result}")
        return result

    def chat(self, message,history):
        """Main chat function that processes queries and returns responses"""
        clear_log_file(log_file)

        new_history= list(history) if history else [] 

        response = requests.get(
            f"http://{HOST}:{PORT}/process", 
            params={
                "prompt": message,
                "collection_name_global": self.collection_name_global
            }, 
        )
        # response = {"response_markdown": response.text}
        # markdown_content = response["response_markdown"]
        json_response = json.loads(response.text)
        markdown_content = json_response["response_markdown"]
        logger.debug(type(markdown_content))
        logger.debug(markdown_content)
        new_history.append((message,markdown_content))
        return new_history

    def think(self, query):
        """Stream the bot's thinking process"""
        self.current_query = query 
        thoughts = []
        # Stream each thought with a small delay
        thinking_text = ""
        for thought in thoughts:
            thinking_text += thought + "\n"
            yield thinking_text    
    

# Define custom CSS
css = """
body {
    background-size: cover;
}
.gradio-container {
    position: relative;
    height: 100vh;
    width: 100vw;
    background : url('/Users/nisarg/Downloads/pexels-jon-champaigne-622172690-27681652.jpg') no-repeat center center fixed;

}
.thinking-box {
    min-height: 400px !important;
    font-family: monospace;
    white-space: pre-wrap;>
    background-image: linear-gradient(to right, #131c2c, #2628aa);
    color: white !important;
}
.header-container {
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    padding: 1rem;
}
.logo {
    position: absolute;
    left: 0rem;
    height: 40px;
    width: 40px;
    align-items: center;
    justify-content: center;
}
.title {
    font-size: 36px;
    font-weight: bold;
    margin: 0;
    padding: 0;
    text-align: left;
}
.input-container {
    display: flex;
    align-items: flex-start;
    color: white !important;
}
.upload-file {
    display: block;
    align-items: flex-start;
}
.submit-btn {
    min-height: 45px;
    background-image: linear-gradient(to right, #192058, #2628aa);
    color: white !important;
}
.search-btn {
    min-height: 45px;
    background-image: linear-gradient(to right, #1f2481, #2628aa);
    color: white !important;
}
.team-73 {
    color: blue !important;
}
.query-box {
    background-image: linear-gradient(to right,#192058, #2628aa);
    color: white !important;
}
.status {
    background-image: linear-gradient(to right, #131c2c, #2628aa);
}
"""

def clear_textbox(msg):
    return ""

# Gradio UI
def create_interface():
    clear_log_file(log_file)
    bot = ChatBot()

    # Define interface
    with gr.Blocks(css=css, title='Pathway', fill_height=True, fill_width=True, theme=gr.themes.Soft()) as interface:
        # Header with logo and title
        with gr.Row(elem_classes=["header-container"]):
            with gr.Column(elem_classes=["header-container"], scale=5):
                gr.Image(
                    show_download_button=False,
                    show_fullscreen_button=False,
                    value="./assets/pathway_new.png",
                    elem_classes=["logo"],
                    show_label=False,
                    container=False
                )
            with gr.Column(elem_classes=["header-container"], scale=6):
                gr.Markdown(value = "# Team 73", elem_classes=["team-73"])

        with gr.Row(height=400):
            with gr.Column(scale=3):
                with gr.Row():
                    gr.Markdown("## Dynamic-Agentic-RAG", elem_classes=["title"])

                with gr.Row():    
                    chatbot = gr.Chatbot(
                        value=[],
                        #elem_id="chatbot",
                        height=400,
                        label="Chat Interface",
                        latex_delimiters=[{ "left": "$$", "right": "$$", "display": True }],
                    )

                with gr.Row(elem_classes=["input-container"]):
                    with gr.Column(scale=1, show_progress=True):
                        with gr.Row(elem_classes=["upload-file"], min_height=100):
                            link_upload = gr.Textbox(
                                show_label=False,
                                placeholder="Enter the google drive link",
                                lines=1,
                                #elem_id="msg-textbox",
                            )

                            file_upload = gr.File(
                                file_types=['.pdf','.json'], 
                                label="Upload your google drive credentials",
                                height=50,
                                scale=0.5,
                                visible=False
                            ) 
                           
                            upload_status = gr.Textbox( 
                                show_label=False, 
                                interactive=False,
                                container=False
                                
                            )
                        # # search_btn = gr.Button("Web Search", elem_classes=["search-btn"], visible=False)                
                
                    with gr.Column(scale=3):
                        with gr.Row(elem_classes=["input-container"]):
                            msg = gr.Textbox(
                                #label="Type your query here...",
                                show_label=False,
                                placeholder="Enter your query and press submit",
                                lines=12,
                                #elem_id="msg-textbox",
                            )

                        with gr.Row(elem_classes=["input-container"]):
                            submit = gr.Button(
                                "Submit", 
                                elem_classes=["submit-btn"]
                                )
                            search_btn = gr.Button(
                                "Web Search", 
                                elem_classes=["search-btn"], 
                                visible=False
                            )

            with gr.Column(scale=2):
                with gr.Row(elem_classes=["title"]):
                    gr.Markdown(
                        value="## Thinking Space"
                    )

                with gr.Row(elem_classes=["thinking-box"]):    
                    gr.Markdown(
                        height=750,
                        value=log_read,
                        label="Thinking Process",
                        every=2,
                        container=True,
                        elem_classes=["thinking-box"]
                    )
                # thinking_box = gr.Markdown(
                #     value=Log(
                #         log_file, 
                #         label="Thinking Process", 
                #         dark=False, 
                #         xterm_font_size=14, 
                #         height=663,
                #         tail=3000
                #     ),
                #     label="Thinking Process",
                #     lines=29,
                #     container=True,
                #     elem_classes=["thinking-box"],
                #     visible=False
                # )

        def perform_web_search(chatbot_state):
            # Get the last query from the chat history
            # if chatbot_state and chatbot_state[-1]:
            last_query = chatbot_state[-1][0]
            search_results = bot.web_search(last_query)
            return chatbot_state + [(last_query, search_results)]        

        submit.click(
                fn=bot.chat,
                inputs=[msg,chatbot],
                outputs=[chatbot],
            ).then(
                fn=clear_textbox,
                inputs=[msg],
                outputs=[msg]
            ).then(
                fn=clear_textbox,
                inputs=[link_upload],
                outputs=[link_upload]
            ).then(
                fn=make_visible,
                outputs=[search_btn]
            )
        
        file_upload.upload(
                fn=bot.upload_fastapi,
                inputs=[file_upload, link_upload],
                outputs=[upload_status]
            )
        
        link_upload.submit(fn=make_visible,
                           outputs=[file_upload])
        
        # Web search button handler
        search_btn.click(
            fn=perform_web_search,
            inputs=[chatbot],
            outputs=[chatbot]
        )

        return interface
    
demo=create_interface()
# Run Gradio
demo.load(show_progress=True)
demo.launch(debug=True, server_name=HOST)