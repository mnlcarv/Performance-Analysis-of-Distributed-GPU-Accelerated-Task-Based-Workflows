--how to use it SELECT import_csv_to_table('/path/to/your/file1.csv', 'table1');

-- CREATE TABLES
CREATE OR REPLACE PROCEDURE LOAD_TABLES(schema text)
LANGUAGE plpgsql
AS $BODY$
BEGIN

    SELECT import_csv_to_table('../raw_data/' || schema || '/DEVICE.csv', 'DEVICE');
    SELECT import_csv_to_table('../raw_data/' || schema || '/ALGORITHM.csv', 'ALGORITHM');
    SELECT import_csv_to_table('../raw_data/' || schema || '/FUNCTION.csv', 'FUNCTION');
    SELECT import_csv_to_table('../raw_data/' || schema || '/CONFIGURATION.csv', 'CONFIGURATION');
    SELECT import_csv_to_table('../raw_data/' || schema || '/RESOURCE.csv', 'RESOURCE');
    SELECT import_csv_to_table('../raw_data/' || schema || '/DATASET.csv', 'DATASET');
    SELECT import_csv_to_table('../raw_data/' || schema || '/PARAMETER_TYPE.csv', 'PARAMETER_TYPE');
    SELECT import_csv_to_table('../raw_data/' || schema || '/PARAMETER.csv', 'PARAMETER');
    SELECT import_csv_to_table('../raw_data/' || schema || '/EXPERIMENT_RAW.csv', 'EXPERIMENT_RAW');

END; 
$BODY$;

-- Create a function to import CSV files into PostgreSQL tables
CREATE OR REPLACE FUNCTION import_csv_to_table(csv_file_path text, table_name text)
RETURNS void AS $$
DECLARE
    sql_statement text;
BEGIN
    -- Dynamically create the table using the CSV header
    EXECUTE 'CREATE TABLE IF NOT EXISTS ' || table_name || ' AS SELECT * FROM ' || 'csv_to_table(''' || csv_file_path || ''')';

    -- Import CSV data into the created table
    sql_statement := 'COPY ' || table_name || ' FROM ''' || csv_file_path || ''' WITH CSV HEADER';
    EXECUTE sql_statement;
END;
$$ LANGUAGE plpgsql;

-- Function to read CSV header and create a table dynamically
CREATE OR REPLACE FUNCTION csv_to_table(csv_file_path text)
RETURNS void AS $$
DECLARE
    table_name text;
    header text;
BEGIN
    -- Extract table name from CSV file path
    table_name := substring(csv_file_path from '([^/]+)\.csv$');

    -- Read the CSV header
    EXECUTE 'COPY (SELECT * FROM ' || csv_file_path || ') TO STDOUT WITH CSV HEADER' INTO header;

    -- Dynamically create the table using the CSV header
    EXECUTE 'CREATE TABLE IF NOT EXISTS ' || table_name || ' (' || header || ')';
END;
$$ LANGUAGE plpgsql;
