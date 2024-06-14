#-------------------------------------------------------------------
# IMPORTS
#-------------------------------------------------------------------
import flet as ft
import pandas as pd
import json
import sqlite3

#-------------------------------------------------------------------
# CLASS CALCULATO
#------------------------------------------------------------------- 
class Calculator:
    '''
    This class defines a very simple application with a command line
    user interface along with a flet frontend. The class provides
    some basic functions to store and get data, to do some addition
    and subtraction. 

    Instance Attributes:
        txt_filename (str)
            The filename for the json file storing key:value pairs
        db_filename (str)
            The filename for the database storing command history
    '''
    
    #---------------------------------------------------------------
    # __INIT__
    #--------------------------------------------------------------- 
    def __init__(self, page: ft.Page, txt_filename, db_filename):
        '''
        Constructor initializes the class when creating a new
        instance. Here we setup the storage dictionary and dataframe
        and storage database and file. Further more we use the
        txt_filename to grab key:value pairs and the db_filename to 
        get command history

        Args:
            txt_filename : str 
                Filename that stores the key:value pairs
                that the user may have entered previously
            
            db_filename : str
                Filename pointing to sqlite database. This database
                stores the command history
        '''
        # Dictionary to store some key:value pairs
        self.store_dict = {}

        # Pandas dataframe 
        self.store_df = None

        # Get the file name to store text data
        self.txt_filename = txt_filename 

        # Get the file name to store database 
        self.db_filename = db_filename

        # Restore any stored key:value pair information 
        # from json text file
        try:
            with open(self.txt_filename, 'r') as json_file:
                self.store_dict = json.load(json_file)
        except Exception as e:
            print (f'No json file detected: {e}')

        try:
            # Restore any command history from database file
            conn = sqlite3.connect(self.db_filename)

            # Read the data back into a DataFrame
            self.store_df = pd.read_sql('SELECT * FROM history', conn)

            # Close the connection
            conn.close()
        except Exception as e:
            print (f'No database detected: {e}')

    #---------------------------------------------------------------
    # CREATE_INTERFACE
    #--------------------------------------------------------------- 
    def create_interface (self, page):
        
        pass

    #---------------------------------------------------------------
    # ADD_LIST
    #--------------------------------------------------------------- 
    def add_list (self, vals):
        '''
        Adds together the numbers in a list of values.

        Args:
            vals : list 
                This is a list of numbers

        Returns:
            Returns the sum of the values in vals list
        '''
        return sum(vals)

    #---------------------------------------------------------------
    # SUBTRACT_LIST
    #---------------------------------------------------------------
    def subtract_list (self, vals):
        '''
        Subtracts the numbers in a list of values.

        Args:
            vals : list 
                This is a list of numbers

        Returns:
            Returns the subtracted result of values in vals list
        '''
        result = 0 

        # Iterate through the values and subtract
        for i in vals:
            result = result - i

        # Return the results
        return result

    #---------------------------------------------------------------
    # ADD
    #---------------------------------------------------------------
    def add (self, a, b):
        '''
        Adds two numbers together

        Args:
            a : int
                First number

            b : int
                Second number

        Returns:
            Returns the sum of the two numbers a and b
        '''
        return a+b

    #---------------------------------------------------------------
    # STORE
    #---------------------------------------------------------------
    def store (self, a,b):
        '''
        Adds the key:value pair assigned by a:b to a global dictionary
        stored in a global variable 

        Args:
            a : str 
                A key as a string
            
            b : str
                A value to be associated with the key for storage
        '''
        # Store into instance variable
        self.store_dict[a]=b

        # Write the dictionary to a JSON file
        with open(self.txt_filename, 'w') as json_file:
            json.dump(self.store_dict, json_file, indent=4)

    #---------------------------------------------------------------
    # STORE_COMMAND
    #---------------------------------------------------------------
    def store_command (self, cmd, params=[]):
        '''
        Stores the command that the user entered in the command line
        along with its parameters

        Args:
            cmd : str
                Stores the name of the command typed in by the user
            params : list of str 
                Stores the list of parameters used by the user
        '''
        # Convert each integer to a string
        params_str = map(str, params)
        
        # Creates a list out of command and params
        new_row = [cmd, ', '.join(params_str)]

        # Convert the list to a dataframe
        new_row_df = pd.DataFrame([new_row], columns=['Command', 'Params'])

        # Append to the end of the storage dataframe 
        self.store_df = pd.concat([self.store_df, new_row_df], ignore_index =True)

        # Store the dataframe row into database
        conn = sqlite3.connect(self.db_filename)

        # Write the DataFrame row to the SQLite database table called history
        # Replace row if it exists
        new_row_df.to_sql('history', conn, if_exists='append', index=False)

        # Commit and close the connection
        conn.commit()
        conn.close()

    #---------------------------------------------------------------
    # LIST_COMMANDS
    #---------------------------------------------------------------
    def list_commands(self):
        '''
        Prints a list of commands, a history, as entered by the 
        user. 
        '''
        # Reverse the index order of the dataframe
        reversed_df = self.store_df.iloc[::-1]

        # Making sure that all rows and columns are displayed in the terminal
        pd.set_option('display.max_rows', None)  
        pd.set_option('display.max_columns', None)

        # Print out the dataframe in a pretty way with the most
        # recent command on top
        print(reversed_df.to_string())

    #---------------------------------------------------------------
    # GET
    #---------------------------------------------------------------
    def get (self, a):
        '''
        Gets a value for the provided key from the storage dictionary

        Args:
            a : str
                A key for which to retrieve the value from.

        Returns:
            Returns the value associated with the key passed into the
            function. 
        '''
        return self.store_dict[a]

    #---------------------------------------------------------------
    # GETKEYS
    #---------------------------------------------------------------
    def getkeys(self):
        '''
        Grabs the list of all key values stored in the storage
        dictionary

        Returns:
            Returns all the stored keys as a comma separated list
        '''
        keys = list(self.store_dict.keys())
        keys = ', '.join(keys)
        return keys

    #---------------------------------------------------------------
    # SET_VALUE
    #---------------------------------------------------------------
    def set_value(self):
        '''
        Converts the stored values into a set 

        Returns:
            Returns a list of comma separated values containing
            the elements of the set
        '''
        # Convert the list of key values into a set
        lst = self.store_dict.values()
        lst_set = set(lst)
        return lst_set 

    #---------------------------------------------------------------
    # TUPLE_VALUE
    #---------------------------------------------------------------
    def tuple_value(self):
        '''
        Converts the stored values into a tuple of values and returns
        it as a comma separated string

        Returns:
            Returns a comma separted string of tuples
        '''
        # Convert the list of key values into a tuple
        lst = self.store_dict.values()
        lst_tuple = tuple(lst)
        return lst_tuple

    #---------------------------------------------------------------
    # CREATEDB
    #---------------------------------------------------------------
    def createdb(self):
        '''
        Creates a new database to store your command history
        '''

        pass

    #---------------------------------------------------------------
    # HELLO
    #---------------------------------------------------------------
    def hello(self):
        '''
        Prints a starting text for user
        '''
        # Just a hello world
        print ('Hello World')

#---------------------------------------------------------------
# MAIN
#---------------------------------------------------------------
def main():
    '''
    This main function implements the main function of the user
    interface. This involves printing a welcome message and then 
    waiting for user command input. This command input then 
    calls specific functions in the program. 
    '''
    # Creates an instance of a calculator class
    c=Calculator('test.json', 'test.db')

    # Calls the hello function
    c.hello()

    # Start with an empty user input
    user_input = ""
 
    # Create a lambda function for subtract
    subtract = lambda a, b: a - b

    # Continuously repeat
    while not (user_input=="exit"):
        # Get user input
        user_input = input("Enter something: ")

        # Print out the input value
        print(f"You entered: {user_input}")

        try:
            # Add individual number command
            if (user_input== "add"):
                a = input ("Enter first number: ")
                b = input ("Enter second number: ")
                out=c.add(int(a),int(b))
                print (f"{a}+{b}={out}")
                
                # Stores the command entered by the user
                c.store_command(user_input, [a,b])

            # Subtract individual number command
            elif (user_input == "subtract"):
                a = input ("Enter first number: ")
                b = input ("Enter second number: ")
                out = subtract(int(a),int(b))
                print (f"{a}-{b}={out}")

                # Stores the command entered by the user
                c.store_command(user_input, [a,b])

            # Add a list of numbers command
            elif (user_input == "add_list"):
                a = input("Enter a list of values separated by comma: ")
                listvals = a.split(",")
                listvals = [int(v.strip()) for v in listvals]
                out = c.add_list(listvals)
                print (f"sum of {a}={out}")

                # Stores the command entered by the user
                c.store_command(user_input, listvals)

            # Subtract a list of numbers command
            elif (user_input == "subtract_list"):
                a = input("Enter a list of values separated by comma: ")
                listvals = a.split(",")
                listvals = [int(v.strip()) for v in listvals]
                out = c.subtract_list(listvals)
                print (f"subtraction of {a}={out}")

                # Stores the command entered by the user
                c.store_command(user_input, listvals)

            # Store a key:value pair command
            elif (user_input == "store"):
                a = input ("Enter name: ")
                b = input ("Enter value: ")
                c.store(a,b)
                print (f'stored {a}:{b}')

                # Stores the command entered by the user
                c.store_command(user_input, [a,b])

            # Retrieve a value for a given key
            elif (user_input == "get"):
                a = input ("Enter name: ")
                b = c.get(a)
                print (f"Value {a}={b}")

                # Stores the command entered by the user
                c.store_command(user_input, [a,b])

            # Retrieve all keys stored 
            elif (user_input == "getkeys"):
                b = c.getkeys()
                print (f"Stored keys: {b}")

                # Stores the command entered by the user
                c.store_command(user_input)

            # Create a set out of the values stored
            elif (user_input == "set"):
                a = c.set_value()
                print (f"Set of values: {a}")

                # Stores the command entered by the user
                c.store_command(user_input)

            # Create a tuple out of the values stored
            elif (user_input == "tuple"):
                a = c.tuple_value()
                print (f"Tuple of values: {a}")

                # Stores the command entered by the user
                c.store_command(user_input, a)

            # List the commands used in the past
            elif (user_input == "list"):
                c.list_commands()

        # Catch exception and print error message
        except Exception as e:
            print (f'There was an input error: {e}')




if __name__ == "__main__":
    main()    