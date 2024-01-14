import psycopg2
from configparser import ConfigParser
 
def config(filename='db_config/database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)
 
    # get section, default to postgresql
    db = {}
    
    # Checks to see if section (postgresql) parser exists
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
         
    # Returns an error if a parameter is called that is not listed in the initialization file
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
 
    return db

def open_connection(schema_name):
    try:
        # Obtain the configuration parameters
        params = config()
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**params)

        # Open a cursor to perform database operations
        cur = conn.cursor()

        # Example query with parameters
        query = "SET search_path = %s;"
        
        # user_name and schema_name parameters
        parameters = (schema_name)

        # Set database user and schema
        cur.execute(query, parameters)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    return cur, conn
    
def close_connection(cur, conn):
    # Close the cursor and connection
    cur.close()
    conn.close()