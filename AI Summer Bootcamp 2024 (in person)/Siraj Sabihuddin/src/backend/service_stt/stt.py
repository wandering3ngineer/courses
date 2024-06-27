#-----------------------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------------------
from fastapi import FastAPI, Request
from pydub import AudioSegment
import pyaudio
import io
from vosk import Model, KaldiRecognizer
from vosk import SetLogLevel                    

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

@app.post("/audio")
async def audio(request: Request):
    # Read the raw bytes from the request body
    contents = await request.body()

    # Use io.BytesIO to create a file-like object
    audio_file = io.BytesIO(contents)

    # Create a PyAudio file type
    p = pyaudio.PyAudio()

    # Open the audio file using pydub
    audio = AudioSegment.from_file(audio_file)

    # Convert pydub AudioSegment to raw audio data
    raw_audio_data = audio.raw_data
    sample_width = audio.sample_width
    frame_rate = audio.frame_rate
    channels = audio.channels

    # Set up PyAudio stream
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=frame_rate,
                    output=True)

    # Write the raw audio data to the stream
    stream.write(raw_audio_data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio object
    p.terminate()

    return {"message": "Audio file processed successfully"}



# def extract(audioframes=None, model="google"):
#     '''
#     This code takes an input audio file and transcribes text data from it into another file. It makes use
#     of the model provided to do this transcription. 

#     Parameters:
#     ----------
#     audio (pyaudio):
#         An instance of the pyaudio object containing the audio

#     model (str):
#         File path to the model directory for use for speech to text conversion. Currently supported
#         - "google": for Google speech to text via speech recognition library
#         - "../models/vosk****": for VOSK speech to text libraries. 
#     '''
#     transcription=""

#     # If the audioframes variable is None or doesn't contain any data
#     # then quit out of this function with an empty transcription
#     if not isinstance(audioframes, list) or not (len(audioframes) > 0):
#         return transcription

#     # Compute the number of bytes in a frame of audio
#     # Join the audio frame raw bytes together into a single byte string
#     frame_bytes = len(audioframes[0])/param["chunk"]
#     audiodata = b''.join(audioframes)

#     if (model=='google'):
#         # Create a new recognizer object
#         r = sr.Recognizer()
          
#         # Open a recorded audio file from using the frames
#         audio = sr.AudioData(audiodata, param["rate"], frame_bytes)

#         # Transcribe audio to text
#         try:
#             transcription=r.recognize_google(audio)
#         except Exception as e:
#             print(f"No speech was recognized, or processing failed: {e}")
#     else:
#         # Disable VOSK logs 
#         SetLogLevel(-1)

#         # Load Vosk model
#         model_ = Model(model)

#         # Create a recognizer object
#         rec = KaldiRecognizer(model_, param['rate'])
#         rec.SetWords(True)  # To get words along with recognition results

#         # Recognize speech
#         # Process the concatenated audio data
#         if rec.AcceptWaveform(audiodata):
#             result = json.loads(rec.Result())
#             transcription=result.get('text', '')
#         else:
#             print("No speech was recognized, or processing failed.")

#     # Print and return the transcribed audio
#     print (transcription)
#     return transcription

#-----------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------
def main():
    '''
    This is the main function executes the 
    '''
    global config

    # Load the configuration fille to get running parameters
    with open("api.json") as json_file:
        config = json.load(json_file)
    
    # Load the table of conversation histories from the 
    # SQLite database
    historyLoad()

    # Run the LLM model server as a microservice
    try:
        run(host=config["api_host"], port=config["api_port"])
    except KeyboardInterrupt as e:
        logger.info ('Terminating server')


if __name__ == "__main__":
    main()