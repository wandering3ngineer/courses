'''
Contains code for doing text to speech conversion. This code
runs as a server process. The server accepts RESTful commands for
conversion according to url routes specified for the relevant functions.
The resulting response is a StreamingResponse containing audio that
can be written to file

Author: 
    Siraj Sabihuddin

Date: 
    June 28, 2024
'''
#-----------------------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import os
import uvicorn
import io
import importlib
from gtts import gTTS
from pydub import AudioSegment
import pyttsx3
import logging

#-----------------------------------------------------------------------
# LOGGER CONFIG
#-----------------------------------------------------------------------
logging.basicConfig(
    # Set the logging level to DEBUG
    level=logging.DEBUG,         
    # Define the log message format
    format='%(levelname)s: (%(name)s) (%(asctime)s): %(message)s',
    # Define the date format 
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        # Log messages to a file
        logging.FileHandler('tts.log'),
        # Log messages to the console
        logging.StreamHandler()  
    ]
)

# Create a logger object
logger = logging.getLogger("TTS")

#-----------------------------------------------------------------------
# FAST API APP
#-----------------------------------------------------------------------
# Generate the fastapi app object
app = FastAPI()

#-----------------------------------------------------------------------
# CONFIGURATIONS
#-----------------------------------------------------------------------
# Setup global configurations
config_file='tts.json'

#-----------------------------------------------------------------------
# TEXT
#-----------------------------------------------------------------------
@app.post("/text/{model}/{text}")
async def text(model, text):
    '''
    Receives a request for converting some text via some tts model
    into speech.

    Route:
        @app.post("/text/{model}/{text}")

    Args:
        text : str = 
            The text data to convert to speech

        model : str =
            Choose a text to speech engine. Engines available are as follows:
            - gtts: Google Text to Speech engine (cloud based)
            - pyttsx3: Simple python local text to speech engine

    Return:
        audio_bytes: StreamingResponse =
            A byte stream binary object representing the audio 
            data produced as a StreamingResponse object from fastapi
    '''
    # Run the text to speech conversion 
    audio_bytes = speech(text,model)

    if (audio_bytes):
        # This is used by fast api for streaming back large chunks of data. It 
        # allows for better memory management than sending back a response
        # that is just the direct audio_bytes buffer. 
        return StreamingResponse(io.BytesIO(audio_bytes.getvalue()), media_type="audio/wav")
    else:
        return None

#-----------------------------------------------------------------------
# SPEECH
#-----------------------------------------------------------------------
def speech(text, model, language='en', accent='uk'):
    '''
    Generates speech from text input. Uses one of multiple models for speech synthesis. User can choose the engine
    that they wish to use. 

    Args:
        text : str =
            The text data to convert to speech

        model : str =
            Choose a text to speech engine. Engines available are as follows:
            - gtts: Google Text to Speech engine (cloud based)
            - pyttsx3: Simple python local text to speech engine

        language : str =
            Choose the language for the response. Currently only one supported:
            - en: English

        accent : str =
            Choose the accent for the language. Currently only one supported:
            - uk: British

    Return:
        audio_bytes: io.BytesIO = 
            A byte stream binary object representing the audio 
            data produced.
    '''
    try:
        # provide a data structure containing 
        # a mapping for various regional accents
        # to be used by gtts
        ac = {
            'gtts': {
                'en': {
                    'au': 'com.au',
                    'uk': 'co.uk',
                    'us': 'com',
                    'ca': 'ca',
                    'in': 'co.in',
                    'ie': 'ie',
                    'za': 'co.za',
                }
            },
            'pyttsx3': {
                'en': {
                    'uk': 'english_rp',
                    'us': 'english-us',
                }
            }
        }

        # Create a BytesIO object to store the audio
        audio_buffer = io.BytesIO()

        # Use the GTTS engine
        if model == 'gtts':
            # Call google text to speech and save audio to mp3
            gtts = gTTS(text=text, lang=language, tld=ac[model][language][accent], slow=False)

            # Save the speech to the buffer
            gtts.write_to_fp(audio_buffer)
            
            # Move to the beginning of the byte buffer
            audio_buffer.seek(0)

        # Use the PYTTSX3 engine    
        else:  # (model=='pyttsx3'):
            # Initialize the text to speech engine
            importlib.reload(pyttsx3)
            pyttsx = pyttsx3.init()
            for voice in pyttsx.getProperty('voices'):
                if ac[model][language][accent] in str(voice.name):
                    pyttsx.setProperty('voice', voice.id)

            temp_filename = 'tts.wav'
            pyttsx.save_to_file(text, temp_filename)
            pyttsx.runAndWait()
            pyttsx.stop()

            wav_audio = AudioSegment.from_file(temp_filename, format="wav")
            wav_audio.export(audio_buffer, format="wav")
            audio_buffer.seek(0)

        return audio_buffer

    except Exception as e:
        logger.error(f'Failed to generate audio response: {e}')
        return None

#-----------------------------------------------------------------------
# CONFIGLOAD
#-----------------------------------------------------------------------
def configLoad(config_file):
    '''
    Grab the stored config data in the config file

    Args:
        config_file : str =
            The path to the JSON config file

    Returns:   
        config : dict =
            Json dictionary of values from json file
    '''
    # Load the configuration fille to get running parameters
    with open(config_file) as json_file:
        config = json.load(json_file)
    
    # Return dictionary
    return config

#-----------------------------------------------------------------------
# CONFIGSTORE
#-----------------------------------------------------------------------
def configStore (config_file, config_data):
    '''
    Store the updated config data in the config file

    Args:
        config_file : str = 
            The path to the JSON config file
        
        config : dict = 
            The updated dictionary of configuration
            data
    '''
    # Store the updated config into file
    with open(config_file, 'w') as json_file:
        json.dump(config_data, json_file, indent=4)

#-----------------------------------------------------------------------
# RUN
#-----------------------------------------------------------------------
def run(host, port):
    '''
    This function should be called to start the server side api microservice 
    for the backend. 

    Args:
        host : str = 
            The host ip address passed in as a string

        port : str = 
            The host port passed in as a string 

    Returns:
        process : Popen = 
            Returns an instance of the process in case we need to kill later.
    '''
    uvicorn.run(app, host=host, port=int(port), log_level="info")

#-----------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------
def main():
    '''
    This is the main function runs the service after loading the configuration
    data. 
    '''   
    # Load the configuration data
    config = configLoad(config_file)

    # Run the LLM model server as a microservice
    try:
        run(host=config['api']["host"], port=config['api']["port"])
    except KeyboardInterrupt as e:
        logger.info ('Terminating server')

if __name__ == "__main__":
    main()