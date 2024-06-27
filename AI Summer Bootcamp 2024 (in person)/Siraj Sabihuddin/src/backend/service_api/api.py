#-----------------------------------------------------------------------
# IMPORTS
#-----------------------------------------------------------------------
from fastapi import FastAPI
import sqlite3
import json
import uvicorn
from openai import OpenAI
from io import BytesIO
import pycurl
import time
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
        logging.FileHandler('api.log'),
        # Log messages to the console
        logging.StreamHandler()  
    ]
)

# Create a logger object
logger = logging.getLogger("API")

#-----------------------------------------------------------------------
# FAST API APP
#-----------------------------------------------------------------------
# Setup fast api app
app = FastAPI()

#-----------------------------------------------------------------------
# GLOBALS
#-----------------------------------------------------------------------
# Global store for conversation history and 
# any configuration information. These are temporary
# variables that will not persist. If they crash 
history = []
confg = []

#-----------------------------------------------------------------------
# QUERY
#-----------------------------------------------------------------------
@app.get("/query/{model}/{prompt}")
async def query(prompt, model):
    '''
    Sends text from given request file to an LLM model of your choice (internal or external). 
    A key is used to access the model if needed. The resulting response of the LLM is stored. 
    In the case of local internal models, if the local model is not running an error is thrown. 
    Before this function is called with a local model, that local model should be run using the 
    run() function. 

    query : str
        A query that is structured as a JSON as follows

    Args:
        prompt : str
            The input query text
        model : str
            The model to use. If the the model is a local model specify 
            the path to the local model
        key : str
            The API key if needed. 
        host : str
            The url end point to the API location relevant to querying the model
    
    Returns:
        response : str
            The string response to the query
    '''
    # Load the global variable
    global config, history

    # Record start time
    starttime = time.time()

    # Create the query dictionary
    query = {"role": "user", "content": prompt}

    # Append the most recent prompt to the conversation history
    history.append(query)

    # Store the latest prompt
    historyStore(model=model, max_tokens=config['llm_maxtokens'], role="user", content=prompt)

    # Define the model switchiing data structure
    json_data_model = None

    # Define the chat completions data structure
    # This is a JSON data structure for the query in the followin
    # format: 
    # {
    #     "model": "string",                      // The name of the model you want to use (e.g., "gpt-4").
    #     "messages": [                           // An array of message objects.
    #         {
    #         "role": "system|user|assistant",    // The role of the message author. Options are "system", "user", or "assistant".
    #         "content": "string"                 // The content of the message.
    #         }
    #     ],
    #     "temperature": number,                  // (Optional) Sampling temperature, between 0 and 2.
    #     "top_p": number,                        // (Optional) An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.
    #     "n": integer,                           // (Optional) Number of chat completion choices to generate for each input message.
    #     "stream": boolean,                      // (Optional) If set, partial message deltas will be sent as data-only server-sent events as they become available.
    #     "stop": "string or array",              // (Optional) Up to 4 sequences where the API will stop generating further tokens.
    #     "max_tokens": integer,                  // (Optional) The maximum number of tokens to generate in the chat completion.
    #     "presence_penalty": number,             // (Optional) Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    #     "frequency_penalty": number,            // (Optional) Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, reducing the model's likelihood to repeat the same line verbatim.
    #     "logit_bias": {                         // (Optional) Modify the likelihood of specified tokens appearing in the completion.
    #         "token_id": bias                    // Token ID mapped to its bias value (-100 to 100).
    #     },
    #     "user": "string"                        // (Optional) A unique identifier representing your end-user, which can help OpenAI monitor and detect abuse.
    # }
    json_data_query = '{"model":"' + model + '", "messages":' + json.dumps(history) + ', "max_tokens":' + config["llm_maxtokens"] + '}'
    logger.debug (json_data_query)

    # Create an api endpoint
    api_endpoint_model='http://'+ config["llm_host"] + ':' + config["llm_port"] + '/model/' + model
    api_endpoint_query='http://'+ config["llm_host"] + ':' + config["llm_port"] + '/relay/v1/chat/completions'

    # We are going to try to connect to our LLM service. If for some reason the 
    # connection is refused. We will try to start the LLM and re-attempt
    # a connection. If still, there are problems then we just terminate
    try:
        # Call the llm api to change model
        logger.info (f"Changing model to {model} via {api_endpoint_model}")
        response = request(url=api_endpoint_model, data=json_data_model, method="PUT")

        # Note that changing the model takes time. 

        # Check that the request has been completed successfully
        if (response): 
            logger.info (f"Model has been changed to {model}")
        else:
            logger.info (f"Unable to change model to {model}")

        # Call the llm api to make a request to the model selected
        logger.debug (f"Sending prompt {prompt} via {api_endpoint_query}")
        response = request(api_endpoint_query, data=json_data_query, method="POST")

        # Extract the summary from the OpenAI API response. The response should be
        # in the following format
        # {
        #     "id": "chatcmpl-xxxxxxxxxxxxxxxxxxx",   // Unique identifier for the completion.
        #     "object": "chat.completion",            // The type of object returned.
        #     "created": 1612200000,                  // The timestamp when the completion was created.
        #     "model": "gpt-4",                       // The model used for the completion.
        #     "choices": [                            // List of completion choices.
        #         {
        #             "index": 0,                     // The index of the choice.
        #             "message": {                    // The message object of the choice.
        #                 "role": "assistant",        // The role of the message author (always "assistant" for completions).
        #                 "content": "string"         // The content of the generated completion.
        #             },
        #         "finish_reason": "stop"             // The reason why the completion finished (e.g., "stop", "length", "content_filter").
        #         }
        #     ],
        #     "usage": {                              // The usage statistics for the request.
        #         "prompt_tokens": 10,                // Number of tokens in the input prompt.
        #         "completion_tokens": 50,            // Number of tokens in the completion.
        #         "total_tokens": 60                  // Total number of tokens used (prompt + completion).
        #     }
        # }
        response = response['choices'][0]['message']['content']
        logger.debug(f"Obtained response {response} for prompt")

    except Exception as e:
        # Print an error indicating that there is a problem sending
        # the curl request
        logger.error("Error: " + str(e))

    if (response):
        # Append the most recent prompt to the conversation history
        history.append({
            "role": "system",
            "content": response
        })

    # Store the latest response into the database
    historyStore(model=model, max_tokens = config['llm_maxtokens'], role="system",content=response)

    # End time
    endtime = time.time()
    totaltime = endtime - starttime

    # Print the response and response time
    logger.debug (f"Total time: {totaltime}")
    logger.info (f"Response: {response}")
    return response

#-----------------------------------------------------------------------
# REQUEST
#-----------------------------------------------------------------------
def request(url, data, method):
    '''
    This function creates a http request to the particular url endpoint 
    with request body as in the data variable

    Args:
        url : str
            The endpoint url for the request 
        data : str
            The json request body 
        method : 
            The method to use for the request
            e.g. POST, GET, PUT etc. 
    Returns
        response : str
            The json response body. If no response body
            returns None
    '''
    # Construct the cURL header from my keys
    httpheader=[
        "Content-Type: application/json"
    ]

    try:
        # Just a storage buffer
        buffer = BytesIO()

        # Sends request for getting chat completion 
        crl = pycurl.Curl()

        # The url where the request should be sent
        crl.setopt(crl.URL, url)        

        # The http header                  
        crl.setopt(crl.HTTPHEADER, httpheader)            
        
        if method == "POST":
            # Set to send POST request
            crl.setopt(crl.POST, 1)                           
            # Set POST fields (JSON data)
            if (data):
                crl.setopt(crl.POSTFIELDS, data)  

        elif method == "PUT":
            # Set to send PUT request
            crl.setopt(crl.CUSTOMREQUEST, "PUT")  
            # Set PUT fields (JSON data)    
            if (data):        
                crl.setopt(crl.POSTFIELDS, data)                  

        elif method == "GET":
            # Set to send GET request
            crl.setopt(crl.HTTPGET, 1)                        

        # Buffer to receive responses
        crl.setopt(crl.WRITEDATA, buffer)                 
        
        # Perform the request
        crl.perform()                                   

        # Get the HTTP response status code
        http_response_code = crl.getinfo(pycurl.HTTP_CODE)

        # Get and print the response body
        response_body = json.loads(buffer.getvalue().decode('utf-8'))

        # Resets the pycurl instance for next request
        crl.reset()

        # Closes the request connection
        crl.close()    

        # Resets the buffers. 
        buffer.flush()
        buffer.seek(0)

        # Return the response
        response = response_body
        return response

    except Exception as e:
        # Print an error indicating that there is a problem sending
        # the curl request
        logger.error(f"Request error: {e}")
        return None

#-----------------------------------------------------------------------
# HISTORYSTORE
#-----------------------------------------------------------------------
def historyStore (model, max_tokens, role, content):
    '''
    Adds a record of the history to the sqlite database on the api 
    server. This can then be retrieved by the user as needed through
    an API call. If no history table or database exists, then one
    is created. This created database should have a history table
    containing two columns: role and content indexed sequentially
    by the order in which the history was recorded. 

    Args:
        model : str
            The name of the model being used. 
        role : str
            The role of the message. Could be either user or system
        content : str
            The content of the message as a sting. 
        max_tokens : int
            The number of available tokens for response used
            in this message
    '''
    # Bring local into global scope
    global config

    # Initialize connector variable
    conn = None
    try:
        # Connect to the database (if it exists)
        # Create one if it doesn't exist
        conn = sqlite3.connect(config['api_db'])

        # Get the cursor for the db
        cursor = conn.cursor()

        # Create an SQL query for creating a table
        query = """ CREATE TABLE IF NOT EXISTS history 
                    (
                        id integer PRIMARY KEY,
                        model text NOT NULL, 
                        max_tokens integer NONT NULL,
                        role text NOT NULL,
                        content text NOT NULL
                    );"""
        
        # Execute the SQL query
        cursor.execute(query)

        logger.debug (f"Model: {model}, Max Tokens: {max_tokens}, Role: {role}, Content: {content}")

        # Add a record to the table that's been created (if any needed to 
        # be created)
        query = """ INSERT INTO history(model, max_tokens, role, content)
                    VALUES(?, ?, ?, ?)"""
        cursor.execute(query, (model, max_tokens, role, content))

        # Commit the transactions
        conn.commit()
    
    # We've violated some sqlite storage process
    except Exception as e:
        logger.error (f"Storage into sqlite DB failed: {e}")

    # Attempt to close the database connection if existing
    finally:
        if (conn):
            # Close and write the database
            conn.close()

#-----------------------------------------------------------------------
# HISTORYLOAD
#-----------------------------------------------------------------------
def historyLoad():
    '''
    Loads the data from the history table in the storage database.
    This history table is used to provide context if the user has
    been having an on-going conversation with the model
    '''
    # Load the global variable
    global history, config

    # Initialize the connector
    conn = None

    try:
        # Connect to the database (if it exists)
        # Create one if it doesn't exist
        conn = sqlite3.connect(config['api_db'])

        # Get the cursor for the db
        cursor = conn.cursor()

        # Execute query to fetch all rows from the 'history' table
        cursor.execute("SELECT role, content FROM history")
        
        # Fetch all rows from the executed query
        rows = cursor.fetchall()
        
        # Get column names from the cursor description
        column_names = [description[0] for description in cursor.description]
        
        # Convert rows to list of dictionaries
        history = [dict(zip(column_names, row)) for row in rows]
    
    except Exception as e:
        logger.error (f"Unable to retreive conversation history: {e}")

    finally:
        # Ensure the connection is closed
        if conn:
            conn.close()

#-----------------------------------------------------------------------
# HISTORYCLEAR
#-----------------------------------------------------------------------
@app.get("/history/clear")
async def historyClear ():
    '''
    This function clears the database history table and any variables
    that are acting to store this history 
    '''
    # Bring local into global scope
    global config, history

    # Connect to the database (if it exists)
    # Create one if it doesn't exist
    conn = sqlite3.connect(config['api_db'])

    # Get the cursor for the db
    cursor = conn.cursor()

    # Execute DELETE statement to erase all rows from the 'history' table
    cursor.execute("DELETE FROM history")
    
    # Commit the transaction
    conn.commit()

    # Close the database
    conn.close()

    # Clear out the history variable
    global history
    history = []

    # Indicate the operation is done
    return True

#-----------------------------------------------------------------------
# HISTORYLIST
#-----------------------------------------------------------------------
@app.get("/history/list")
async def historyList ():
    # Bring in global variables
    global history

    # Return this global list
    return history

#-----------------------------------------------------------------------
# TOKENSMAX
#-----------------------------------------------------------------------
@app.get("/tokens/max/{tokens}")
async def tokensMax (tokens):
    '''
    This function sets the maximum number of allowed tokens to be
    returned by the model. Note that this is not persistent. 
    So there is no change to the original configuration file

    Args:
        tokens : int
            The maximum number of allowed tokens to be used 
            by the model
    
    Returns:
        response : bool
            Returns true if successfully set, otherwise returns
            false
    '''
    global config
    
    # Set the max_tokens for the current model.
    config['llm_maxtokens']=tokens
    
    # Store the updated config into file
    with open("api.json", 'w') as json_file:
        json.dump(config, json_file, indent=4)

#-----------------------------------------------------------------------
# RUN
#-----------------------------------------------------------------------
def run(host, port):
    '''
    This function should be called to start the server side api microservice 
    for the backend. 

    Args:
        host : str
            The host ip address passed in as a string
        port : str
            The host port passed in as a string 

    Returns:
        process : Subprocess.Popen
            Returns an instance of the process in case we need to kill later.
    '''
    uvicorn.run(app, host=host, port=int(port), log_level="info")   

#-----------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------
def main():
    '''
    This is the main function. It grabs the config data, historical data
    for the conversation and stores it in a set of globals. The web
    server is then run and operates until the user uses a keyboard interrupt
    to break the execution. 
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
        