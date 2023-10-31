from functools import lru_cache
import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from utils.common_functions import list_distribute_into_blocks
import json
import logging

# Set up logging
_logs_file_ops = logging.getLogger(__name__)

try:
    # Import necessary libraries
    import numpy as np
    import pandas as pd
    import boto3
    from botocore.errorfactory import ClientError
    s3 = boto3.client('s3')
    s3_resource = boto3.resource('s3')
    from pyarrow.lib import ArrowInvalid
    import pyarrow.parquet as pq
    from tqdm import tqdm
except Exception as e:
    # Log warning if import fails
    _logs_file_ops.warning(f'Failed import: {e}')

@lru_cache(maxsize=None)
def _bucket_prefix(path):
    """Splits the path to an s3 URI into its bucket name and prefix
    """
    bucket_name = re.search(r's3://([^/]+)/', path).group(1)
    prefix = re.search(r's3://[^/]+/(.+?)/?$', path).group(1)
    return bucket_name, prefix

def list_s3(folder_path):
    """Returns the full path to the contents of a specified s3 location
    """
    bucket, prefix = _bucket_prefix(folder_path)
    res = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    res = res.get('Contents', None)
    if res:
        return [f"s3://{bucket}/{_['Key']}" for _ in res]
    else:
        return []

def listdir(folder_path):
    """Returns a list of full paths to the contents of a specified location, accepts s3
    """
    if 's3://' in folder_path:
        folder_path = folder_path.replace('\\', '/')
        # create S3 client
        # specify bucket and prefix (folder path)
        # bucket_name = 'boulevarddataplatform'
        bucket_name, prefix = _bucket_prefix(folder_path)
        if prefix[-1] != '/':
            prefix += '/'

        keys = []
        kwargs = dict(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        while True:
            resp = s3.list_objects_v2(**kwargs)
            files, folders = resp.get('Contents', []), resp.get('CommonPrefixes', [])
            for obj in files:
                keys.append(obj['Key'])
            for obj in folders:
                keys.append(obj['Prefix'])
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break

        # Log number of files and folders found
        _logs_file_ops.info(f'{len(files)} files + {len(folders)} folders in {bucket_name}/{prefix}')
        return [f's3://{bucket_name}/{_}' for _ in keys]
    else:
        return [os.path.join(folder_path, _) for _ in os.listdir(folder_path)]

def touch_file(file_path):
    """Creates an empty file if it does not exist
    """
    if 's3://' in file_path:
        file_path = file_path.replace('\\', '/')
        # s3 = boto3.client('s3')
        # Extract bucket name
        # bucket_name = re.search(r's3://([^/]+)/', file_path).group(1)
        bucket_name, prefix = _bucket_prefix(file_path)

        # Extract key (object key)
        key = re.search(r's3://[^/]+/(.+)', file_path).group(1)
        s3.put_object(
            Bucket=bucket_name,
            Key=key
        )
    else:
        with open(file_path, 'w') as document:
            pass

def touch_folder(folder_path):
    """Creates an empty folder if a folder of the path does not exists
    """
    if 's3' in folder_path:
        return None
    elif not os.path.exists(folder_path):
        os.makedirs(folder_path)

def path_join(*args):
    """Joins paths such as folders and file names into a valid path string
    """
    joined = os.path.join(*args)
    if 's3' in joined:
        return joined.replace('\\', '/')
    else:
        return joined

def read_from_json(path):
    """Reads to a dict a json file if it exists, else will return None
    """
    if 's3' in path:
        bucket, prefix = _bucket_prefix(path)
        content_object = s3_resource.Object(bucket, prefix)
        try:
            file_content = content_object.get()['Body'].read().decode('utf-8')
            json_content = json.loads(file_content)
        except ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                _logs_file_ops.info(f"Key does not exist!")
            else:
                _logs_file_ops.error(f"Error accessing Key: {e.response['Error']}")
            json_content = None

    elif os.path.exists(path):
        with open(path) as f_in:
            json_content = json.load(f_in)

    else:
        json_content = None
    return json_content

def write_to_json(data_dict: list[dict], path):
    """Writes a dict or list[dict] to the specified file as a valid JSON
    """
    if 's3://' in path:
        bucket, prefix = _bucket_prefix(path)
        s3object = s3_resource.Object(bucket, prefix)

        s3object.put(
            Body=(bytes(json.dumps(data_dict).encode('UTF-8')))
        )
    else:
        with open(path, 'w') as f:
            json.dump(data_dict, f, ensure_ascii=False)

def write_file(data, path: str):
    """Writes bytes data to the path (key) provided for s3 or local storage based on the key.
    Do NOT use this to write dicts to JSON. Use `write_to_json` instead

    Args:
        data (bytes): Bytes object
        path (str): Storage file path
    """
    if 's3://' in path:
        bucket, prefix = _bucket_prefix(path)
        s3object = s3_resource.Object(bucket, prefix)
        s3object.put(Body=data)
        # s3.put_object(Body=data, Bucket=bucket, Key=prefix)
    else:
        with open(path, 'wb') as handle:
            handle.write(data)

def read_file(path: str):
    """Reads bytes data to the path (key) provided for s3 or local storage based on the key

    Args:
        path (str): Storage file path
    """
    if not path:
        _logs_file_ops.warning('Path was None')
        return None
    if 's3://' in path:
        bucket, prefix = _bucket_prefix(path)
        s3object = s3_resource.Object(bucket, prefix)
        data = s3object.get()['Body'].read()
        # s3.put_object(Body=data, Bucket=bucket, Key=prefix)
    else:
        with open(path, 'rb') as handle:
            data = handle.read()
    return data

def read_parquets(
        path:str | list[str], columns:list[str] | None = None,
        threads:int=10, max_retries:int = 5, replace_npnan:bool = False
    ):
    """Reads the specified path (s3 or local) as a pandas dataframe

    Args:
        path (str | list[str]): A folder or a list of parquet files
        - If path the function will look for .parquet files and concat them
        - If a list of paths, the function will filter .parquet files and concat them

        columns (list[str] | None, optional): Specify columns to be read from the parquet file
        - Specify columns to reduce the size of data being transfered and stored in memory
        - If `None`, all columns will be read
        
        threads (int, optional): Number of parallel threads reading
        - Do NOT exceed 10 for s3, else you may hit the rate limit
        
        max_retries (int, optional): Max times to retry
        replace_npnan (bool, optional): Replace nan values?

    Returns:
        pandas.DataFrame: A pandas dataframe with all the parquet files concatenated
    """
    # Increase threads to 500 if path is not s3
    if not any(['s3:' in path, any(['s3:' in _ for _ in path])]):
        threads=500

    def _read_parquet_file(file_path, columns=None, retry_number = 0):
        """Read a single parquet file

        Args:
            file_path (str): Path to the parquet file
            columns (list[str] | None, optional): Specify columns to be read from the parquet file
            retry_number (int, optional): Number of times the function has retried

        Returns:
            pandas.DataFrame | None: A pandas dataframe with the contents of the parquet file, or None if the file could not be read
        """
        # Return None if max retries have been exceeded
        if retry_number >= max_retries:
            _logs_file_ops.error(f'Failed for {file_path} after {retry_number} retries')
            return None
        try:
            # Read the parquet file
            parquet_file  = pq.ParquetFile(file_path)
            # Return None if the file is empty
            if parquet_file.metadata.num_rows == 0:
                return None
            # Filter columns if specified
            if columns is not None:
                columns = [_ for _ in columns if _ in parquet_file.schema_arrow.names]
                if len(columns)==0: return None
            # Return the contents of the parquet file as a pandas dataframe
            return (parquet_file.read(columns=columns).to_pandas())
        except Exception as e:
            # Retry if an exception is raised
            if isinstance(e, ArrowInvalid):
                _logs_file_ops.warning(f'Skipping invalid parquet file: {file_path}')
                return None
            _logs_file_ops.warning(f'Retrying: {file_path}: {e}')
            # Full jitter wait
            temp = min(10, 2**retry_number)
            wait_time = temp/2 + random.uniform(0, temp)
            time.sleep(wait_time)
            # Retry
            return _read_parquet_file(file_path, columns, retry_number+1)

    def _read_list_of_parquet_files(file_list: list[str], columns=None):
        """Read a list of parquet files serially using `_read_parquet_file`

        Args:
            file_list (list[str]): List of paths to parquet files
            columns (list[str] | None, optional): Specify columns to be read from the parquet file

        Returns:
            pandas.DataFrame: A pandas dataframe with the contents of all the parquet files concatenated
        """
        # Concatenate the contents of all the parquet files in the list
        df = pd.concat([
                _read_parquet_file(file_path=f, columns=columns)
                for f in tqdm(file_list, desc = 'Reading...', leave = False)
            ])
        # Replace np.nan with replace_npnan if specified
        if replace_npnan is False:
            return df
        else:
            _logs_file_ops.warning(f'[!] Replace np.nan with {replace_npnan}')
            return df.replace(np.nan, replace_npnan)

    # Determine if path is a string or a list
    if isinstance(path, str):
        if '.parquet' in path:
            # Read a single parquet file
            return pd.read_parquet(path=path, columns=columns)
        else:
            # Get a list of all the parquet files in the folder
            list_of_files = [_ for _ in listdir(path) if '.parquet' in _]
    elif isinstance(path, list):
        # Use the provided list of paths to parquet files
        list_of_files = path

    # Return None if no parquet files were found
    if len(list_of_files) == 0:
        _logs_file_ops.warning(f'No .parquet files found in {path}')
        return None

    # Distribute the list of files into blocks for parallel processing
    if threads:
        file_list_of_lists = list_distribute_into_blocks(list_of_files, threads)
    else:
        file_list_of_lists = [[_] for _ in list_of_files]
    # Read the parquet files in parallel using multiple threads
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(_read_list_of_parquet_files, file_list, columns)
            for file_list in file_list_of_lists
        ]
        sublists = [future.result() for future in futures]

    # Concatenate the contents of all the sublists
    return pd.concat(sublists)