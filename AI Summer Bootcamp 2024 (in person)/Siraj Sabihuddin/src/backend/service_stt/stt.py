'''
Contains code for doing spech to text conversion. This code
runs as a server process. The server accepts RESTful commands for
conversion according to url routes specified for the relevant functions
The resulting response is data. 

Author: 
    Siraj Sabihuddin

Date: 
    June 28, 2024
'''
#-----------------------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------------------
from fastapi import FastAPI, Request
from pydub import AudioSegment
import speech_recognition as sr
import io
import vosk  
import json
import uvicorn
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
        logging.FileHandler('stt.log'),
        # Log messages to the console
        logging.StreamHandler()  
    ]
)

# Create a logger object
logger = logging.getLogger("STT")

#-----------------------------------------------------------------------
# FAST API APP
#-----------------------------------------------------------------------
# Generate the fastapi app object
app = FastAPI()

#-----------------------------------------------------------------------
# CONFIGURATIONS
#-----------------------------------------------------------------------
# Setup global configurations
config_file='stt.json'

#-----------------------------------------------------------------------
# AUDIO
#-----------------------------------------------------------------------
@app.post("/audio/{model}")
async def audio(model, request: Request):
    '''
    This code takes an input audio file byte stream and transcribes text 
    data from it. It makes use of the model provided to do this transcription. 

    Route:
        @app.post("/audio/{model}")

    Args:
        model : str =
            File path to the model directory for use for speech to text conversion. Currently supported
            - "google-sr": for Google speech to text via speech recognition library
            - "./models/vosk****": for VOSK speech to text libraries. 

        request : Request =
            A fastapi request object containing the audio 
            data to be converted to text.  

    Returns:
        transcription : str = 
            The text data transcription of the audiofile
    '''
    global config_file

    # Read the raw bytes from the request body
    contents = await request.body()

    # Use io.BytesIO to create a file-like object
    audio_bytes = io.BytesIO(contents)

    # Check the current configuration
    config = configLoad(config_file)

    logger.debug(f"Model: {model}")

    # Get the model index for the model to use
    for m, i in zip(config['model'], range(0,len(config['model']))):
        if (m['name']==model):
            model_index = i

    # Get the model_path for this index
    model_path = config['model'][model_index]['file']
    if (model_path is None): model_path = model

    # Extract the frames from the audio file object
    transcription = transcribe(audio_bytes, model_path)

    # Returns the transcription of the file
    return transcription

#-----------------------------------------------------------------------
# TEXT
#-----------------------------------------------------------------------
def transcribe(audio_bytes, model):
    '''
    This code takes an input audio file byte stream and transcribes text 
    data from it. It makes use of the model provided to do this transcription. 

    Args:
        audio_bytes : byte stream = 
            A byte stream object containing the audio

        model : str = 
            File path to the model directory for use for speech to text conversion. Currently supported
            - "google-sr": for Google speech to text via speech recognition library
            - "./models/vosk****": for VOSK speech to text libraries. 
    
    Returns:
        transcription : str = 
            The text data transcription of the audiofile
    '''
    # Open the audio file using pydub
    audio = AudioSegment.from_file(audio_bytes)

    # Convert pydub AudioSegment to raw audio data
    audio_raw = audio.raw_data
    audio_sample_width = audio.sample_width
    audio_frame_rate = audio.frame_rate
    audio_channels = audio.channels

    # If audio has more than one channel, convert to mono
    if audio_channels > 1:
        audio = audio.set_channels(1)
        audio_raw = audio.raw_data
        audio_channels = 1

    # Create an empty string for the transcription
    transcription=""

    # If the model is 
    if (model=='google-sr'):
        # Create a new recognizer object
        r = sr.Recognizer()
          
        # Open a recorded audio file from using the frames
        audio = sr.AudioData(audio_raw, audio_frame_rate, audio_sample_width)

        # Transcribe audio to text
        try:
            transcription=r.recognize_google(audio)
        except Exception as e:
            logger.error(f"No speech was recognized, or processing failed: {e}")
    else:
        # Disable VOSK logs 
        vosk.SetLogLevel(-1)

        # Load Vosk model
        model_ = vosk.Model(model)

        # Create a recognizer object
        rec = vosk.KaldiRecognizer(model_, audio_frame_rate)
        # To get words along with recognition results
        rec.SetWords(True)  

        # Recognize speech
        # Process the concatenated audio data
        if rec.AcceptWaveform(audio_raw):
            # Load the json data from rec into results
            result = json.loads(rec.Result())

            # Print the json data
            logger.info(f"Results: {result}")

            # If there is a text key in the results then return its value.
            # otherwise return an empty string
            transcription=transcription + result.get('text', '')
        else:
            logger.error("ERROR: No speech was recognized, or processing failed.")

    # Print and return the transcribed audio
    logger.info (transcription)
    return transcription

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