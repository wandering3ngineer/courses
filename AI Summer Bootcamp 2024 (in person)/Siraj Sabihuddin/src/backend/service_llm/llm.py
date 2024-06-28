'''
Contains code for starting and executing various LLMs. On the front
side this code will wait for RESTful requests and then restart
the relevant private LLMs or stop them and use an external LLM such
as openai's gpt-4. To support queries of these individual LLM processes
the code relays requests coming from external source as is if the 
appropriate relay endpoint is provided. It relies on the fact that
all models use openai's RESTful api format.  

Author: 
    Siraj Sabihuddin

Date: 
    June 28, 2024
'''
#-----------------------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------------------
import time
import subprocess
import json
from fastapi import FastAPI, Request, Response
import uvicorn
import httpx
import copy
import os
import signal
import logging

#-----------------------------------------------------------------------
# LOGGER CONFIG
#-----------------------------------------------------------------------
logging.basicConfig(
    # Set the logging level to DEBUG
    level=logging.DEBUG,         
    # Define the log message format
    format='%(levelname)s: (%(name)s[%(funcName)s]) (%(asctime)s): %(message)s',
    # Define the date format 
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        # Log messages to a file
        logging.FileHandler('api.log'),
        # Log messages to the console
        logging.StreamHandler()  
    ]
)

# Create a logger object
logger = logging.getLogger("LLM")

#-----------------------------------------------------------------------
# FAST API APP
#-----------------------------------------------------------------------
# Generate the fastapi app object
app = FastAPI()

#-----------------------------------------------------------------------
# CONFIGURATIONS
#-----------------------------------------------------------------------
# Setup globals for the process objects
# and for configuration information
config_file='llm.json'

#-----------------------------------------------------------------------
# MODEL
#-----------------------------------------------------------------------
@app.put("/model/{model}")
async def model(model):
    '''
    Changes the model to the indicated model value if it exists. This 
    requires that the existing model is stopped and a new model is 
    restarted on the server

    Route: 
        @app.put("/model/{model}")

    Args:
        model : str = 
            The model name to change to. Note that valid model
            names are located in the config file
    '''
    # Strip any extra quotations
    model = model.replace('"', '').replace("'", '')

    # Get the configuration data
    config, model_index = configLoad(config_file)

    # Model name:
    name = config['model'][model_index]['name']

    # Check if the current model is the one that the 
    # user wants. If not then we need to terminate
    # the models and start the one the user wants
    if (name != model):
        # Terminate all models
        config = terminate(config, model=True, api=False)

        # Default response is that no model was found
        response = False

        # Iterate through all models and find the index of the 
        # desired user model. If found, start that model. If not found do
        # nothing
        for m, i in zip(config['model'], range(0,len(config['model']))):
            if (m['name']==model):
                # write a response
                response = True

                # If there is a file associated with this model then run
                # it. Otherwise, its an online model and no further action
                # is needed
                if (m['file']!=None):
                    config['model'][i]['pid']=run_llm(model=config['model'][i]['file'], host=config['model'][i]['host'], port=config['model'][i]['port'], pid=config['model'][i]['pid'])
                    
                    # Notice that changing the model takes time so delay until it is started
                    time.sleep(15)

                # update the config to indicate that the active model has changed
                config['api']['model']=model
    else:
        response = True

    # Store the updated configuration data
    configStore(config_file, config)
    
    # Return the response
    return response

#-----------------------------------------------------------------------
# RELAY
#-----------------------------------------------------------------------
@app.api_route("/relay/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def relay(request: Request, path: str):
    '''
    This function acts as a relay. Basically any request sent to the endpoint
    will be relayed on to either a public or private LLM. 

    Route:
        @app.api_route("/relay/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])

    Args:
        path : str =
            The path after the host:port name

        request : Request = 
            The fastAPI Request object

    Returns:
        response : Response = 
            The FastAPI response object
    '''
    async with httpx.AsyncClient(timeout=900.0) as client:
        # Load the configuration data
        config, model_index = configLoad(config_file)

        # Depending on the key type we will need to either
        # use http or https
        if (config['model'][model_index]['auth']=='sso-key'):
            httptype = "http"
        else:
            httptype ="https"
        
        # Define the endpoint
        api_endpoint = f"{httptype}://{config['model'][model_index]['host']}:{config['model'][model_index]['port']}/{path}"

        try:
            # Modify headers before forwarding to include the api key
            headers = dict(request.headers)
             # Remove the host header
            headers.pop('host', None) 
            # Add the API key to the headers
            headers['Authorization'] = f"{config['model'][model_index]['auth']} {config['model'][model_index]['key']}"
            # Forward the request to the remote service and store the response.
            resp = await client.request(
                method=request.method,
                url=api_endpoint,
                headers=headers,
                content=await request.body()
            )
            
            # Construct and return a FastAPI response object
            response = Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers)
            )
            return response

        except httpx.HTTPStatusError as e:
            logger.error(f"http status error: {e}")
            return Response(
                content=str(e),
                status_code=e.response.status_code,
                headers=dict(e.response.headers)
            )

        except Exception as e:
            logger.error(f"Error: {e}")
            return Response(
                content=str(e),
                status_code=500
            )

#-----------------------------------------------------------------------
# RUN_LLM
#-----------------------------------------------------------------------
def run_llm(model, host, port, pid=None):
    '''
    This function should be called to start the server app microservice 
    for the LLM. The LLM then waits for API requests from another microservice.
    The API format is the OpenAI API format 

    Args:
        model : str = 
            The path to the LLM model of interest to us. 
            This should be a .gguf model file. 

        host : str = 
            The host ip address passed in as a string

        port : str = 
            The host port passed in as a string  

        pid : int = 
            The pid to existing process
            if it exists

    Returns:
        pid : int = 
            Process pid for allowing termination from outside
            the run command
    '''
    try:
        command = [
            'python',
            '-m',
            'llama_cpp.server',
            '--model',
            model,
            '--host',
            host,
            '--port',
            port,
            '--chat_format',
            'chatml',
            '--verbose',
            'False'
        ]

        # Try starting a new LLM process
        logger.info(f"Start a new LLM process: {command}")

        # Make sure any existing processes are stopped
        if (pid is not None):
            logger.info ("Terminating existing LLM processes")
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except Exception as e:
                logger.error (f'PID doesnt exist: {e}')
    
        # Start a OS process running local LLM
        # Run the command
        process = subprocess.Popen(command)     
        
        # Indicate the process PID and sleep for 5 seconds 
        # to give the server time to boot up
        pid = int(process.pid)
        logger.info ("PID of Started LLM Process: " + str(pid))  

    # If something happens with server start up then just print the error    
    except Exception as e:
        logger.error("Error: " + str(e))
        logger.error("Failed to start local LLM and unable to receive request into LLM")      

    # Regardless of failure or success return the process objects
    finally:
        return pid

#-----------------------------------------------------------------------
# RUN_API
#-----------------------------------------------------------------------
def run_api(host, port, pid=None):
    '''
    This function should be called to start the server for the API microservice 
    for the LLM. The API service allows users to change and start a different
    private model or to call an external model such as that of openai 

    Args:
        host : str = 
            The host ip address passed in as a string

        port : str = 
            The host port passed in as a string  

        pid : int = 
            The pid to existing process
            if it exists
    
    Returns:
        pid : int = 
            Process pid for allowing termination from outside
            the run command
    '''
    try:
        command = [
            'uvicorn',
            'llm:app',
            '--host',
            host,
            '--port',
            port
        ]

        # Try starting a new LLM process
        logger.info(f"Start a new API process {command}")

        # Make sure any existing processes are stopped
        if (pid is not None):
            logger.info ("Terminating existing API processes")
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except Exception as e:
                logger.error (f'PID doesnt exist: {e}')

        # Start a OS process running local LLM
        # Run the command
        process = subprocess.Popen(command)     
        
        # Indicate the process PID and sleep for 5 seconds 
        # to give the server time to boot up
        pid = int(process.pid)
        logger.info ("PID of Started API Process: " + str(pid))  

    # If something happens with server start up then just print the error    
    except Exception as e:
        logger.error("Error: " + str(e))
        logger.error("Failed to start API service and unable to receive request")      

    # Regardless of failure or success return the process objects
    finally:
        return pid

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

        model_index : int = 
            The index of the active model
    '''
    # Load the configuration fille to get running parameters
    with open(config_file) as json_file:
        config = json.load(json_file)
    
    # Get current active model being used by api service
    for m, i in zip(config['model'], range(0,len(config['model']))):
        if (m['name']==config['api']['model']):
            model_index = i

    # Return dictionary
    return config, model_index

#-----------------------------------------------------------------------
# CONFIGSTORE
#-----------------------------------------------------------------------
def configStore (config_file, config):
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
        json.dump(config, json_file, indent=4)

#-----------------------------------------------------------------------
# TERMINATE
#-----------------------------------------------------------------------
def terminate (config, model=True, api=True):
    '''
    Terminates the processes in the config data and updates
    the configuration to reflect the new pid values of null

    Args:
        config : dict = 
            The path to the JSON config file
    '''   
    # Kill all model processes
    if (model):
        for i in config['model']:
            if i['pid'] is not None:
                logger.info (f"Terminating process: {i['pid']}")

                # Kill the existing pid
                os.kill(i['pid'], signal.SIGKILL)

                # Wait for it to die
                try: os.waitpid(i['pid'], 0)
                except: pass

                # Update pid to None
                i['pid']=None
    
    # Kill api process
    if (api):
        if config['api']['pid'] is not None:
            logger.info (f"Terminating process: {config['api']['pid']}")

            # Kill the existing pid
            os.kill(config['api']['pid'], signal.SIGKILL)
            
            # Wait for it to die
            try: os.waitpid(config['api']['pid'], 0)
            except: pass
            
            # Update the pid to None
            config['api']['pid']=None

    # Returns the updated config with the update pid values
    return config

#-----------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------
def main(): 
    '''
    This is the main function it executes and starts the uvicorn server 
    for llama-cpp-python and configures the server to run at a specific 
    IP:Port and using a specific private model   
    '''
    global config_file
    config, model_index = configLoad(config_file)

    # Run the LLM model server as a microservice
    pid_llm=run_llm(model=config['model'][model_index]["file"], host=config['model'][model_index]["host"], port=config['model'][model_index]["port"], pid=config['model'][model_index]["pid"])
    pid_api=run_api(host=config['api']["host"], port=config['api']["port"], pid=config['api']['pid'])

    # Store the process objects in a global variable
    config['model'][model_index]['pid'] = pid_llm
    config['api']['pid']= pid_api

    # Store the updated config data
    configStore (config_file, config)

    # Block and run until a keyboard exception occurs and terminate
    try:
        while (True):
            time.sleep(10)
    except KeyboardInterrupt as e:
        # Terminates all processes
        config=terminate(config)

        # Stores the updated configuration back to file
        configStore(config_file, config)

if __name__ == "__main__":
    main()
        
