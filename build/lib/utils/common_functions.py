"""A set of commonly used functions"""

from functools import lru_cache
import re
import logging
import pandas
_logs_common_func = logging.getLogger(__name__)
from timeit import default_timer
default_timer = default_timer

try:
    from pandasgui import show
except:
    pass

def get_uuid():
    """Provides a large unique randomly generated filename friendly ID

    Returns:
        str: the ID
    """
    from uuid import uuid4
    return str(uuid4())

def get_uuid_dated(date: str = None) -> str:
    """Appends the specified date to a randomly generated 8 digit hexadecimal ID. Ideal for use to store files where the name of the file also states the date of creation of the file

    Args:
        date (str | datetime.date, optional): Uses current date if no argument is passed. Defaults to None.

    Returns:
        str: 8-Byte ID appended to today's date in yyyymmdd format
    """
    from datetime import datetime
    if not date:
        date = datetime.today().date()
    return f"{str(date)}_{str(get_uuid().split('-', 1)[0])}".replace('-', '')


def extract_url_parameters(url):
    from urllib.parse import urlparse, parse_qs
    parsed_url = urlparse(url)
    parameters = parse_qs(parsed_url.query)
    parameters = {key:value[0] if len(value) == 1 else value for key, value in parameters.items()}

    return parameters
# Function to create an unordered list from a list with repeating values
def dedupe_list(input_list: list) -> list:
    """A function used to remove duplicates from a list while preserving the order
    in which elements occur in it

    Args:
        input_list (list): A list to be de-duplicated

    Returns:
        list: the input list with duplicates removed
    """
    input_list = list(input_list)
    return list(dict.fromkeys(input_list))

def flatten_list(alist: list) -> list:
    """A function used to reduce a list of lists to a list

    Args:
        alist ([list]): A list of lists

    Returns:
        list: A list
    """
    return [item for sublist in alist for item in sublist]

def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def list_split_into_blocks(input_list: list, block_size: int) -> list[list]:
    """Splits an input list in to a list of lists of specified block size

    Args:
        input_list (list): A python flat list
        block_size (int): Size of blocks

    Returns:
        list[list]: _description_
    """
    if block_size == 0:
        return input_list
    try:
        return [input_list[i:i+block_size] for i in range(0, len(input_list), block_size)]
    except Exception as e:
        _logs_common_func.error(f'Conversion to blocks of list failed due to: {e}')
        return input_list

def list_distribute_into_blocks(input_list: list, num_lists: int = 0) -> list[list]:
    """Equally distributes an input list in to a list of specified number of lists

    Args:
        input_list (list): A python flat list
        block_size (int): Size of blocks

    Returns:
        list[list]: _description_
    """
    if num_lists == 0:
        return input_list
    try:
        list_length = len(input_list)
        sublist_size = list_length // num_lists  # Calculate the size of each sublist
        remaining_elements = list_length % num_lists  # Calculate the number of remaining elements

        sublists = []
        index = 0

        for i in range(num_lists):
            sublist = input_list[index: index + sublist_size]
            index += sublist_size
            # If there are remaining elements, distribute them across sublists
            if remaining_elements > 0:
                sublist.append(input_list[index])
                index += 1
                remaining_elements -= 1

            sublists.append(sublist)

        sublists = [_ for _ in sublists if len(_) > 0]
        return sublists
    except Exception as e:
        _logs_common_func.error(f'Distribution to list of {num_lists} lists failed due to: {e}')
        return input_list

def pickle_to_lzma(raw: str, overwrite = True, delete_original = False) -> bool:
    """Compresses and saves a pickle file with LZMA compression. LZMA has one of the highest compression ratios but is lower on performance

    Args:
        raw (str): Provide the full file path to the pickle file
        overwrite (bool): If True, overwrite the compressed file if it already exists
        delete_original (bool): If True, delete the original file after compression

    Returns:
        bool: Return True if the entire operation succeeded
    """
    import pickle, lzma, os
    success = False

    # Check if the file is already compressed using lzma
    if '.lzma' in raw:
        _logs_common_func.warning(f'Looks like the file is already compressed using lzma: {raw}')
        success = False
        return success

    # Load the data from the pickle file
    try:
        with open(raw, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        _logs_common_func.error(f'Failed to read {raw} due to :{e}')
        success = False
        return success

    # Compress the data using lzma and save it to a new file
    try:
        compressed_file_name = raw + '.lzma'

        if overwrite is False and os.path.isfile(compressed_file_name):
            _logs_common_func.warning(f'Skipped {compressed_file_name}: already exists')
        else:
            with lzma.open(compressed_file_name, "wb") as f:
                pickle.dump(data, f)
        success = True
    except Exception as e:
        _logs_common_func.error(f'Failed to write compressed file to {compressed_file_name} due to :{e}')
        success = False
    finally:
        # Delete the original file if requested and the compression was successful
        if delete_original and success:
            _logs_common_func.warning(f'[!] Deleting {raw}')
            os.remove(raw)
        return success

def column_types(data:pandas.DataFrame, sort_columns:bool = False) -> pandas.DataFrame:
    """Provides a summary of datatypes in a column with the count of the number of each type of datatype for each column

    Args:
        data (pandas.DataFrame): Pass a pandas dataframe

    Returns:
        pandas.DataFrame: Pivot table of column X datatype counts of each row
    """
    from collections import Counter
    metadata = {column: list(data[column].apply(type)) for column in list(data.columns)}
    metadata = {key:dict(Counter([_.__name__ for _ in value])) for key,value in metadata.items()}
    metadata = pandas.DataFrame.from_dict(metadata).fillna(0).astype(int).transpose().rename_axis('column').reset_index()
    if sort_columns:
        metadata.sort_values(by = 'column', inplace=True)
    return metadata

@lru_cache(maxsize=None)
def generate_hash(data: str):
    """Generates a 64bit hash using MurmurHash64, which is an optimized non-cryptographic hash function. It is designed for generating high-quality hash values quickly.

    Args:
        data: Preferably used with str. Will attempt to convert lists and sets to strings
    """
    import mmh3
    if not isinstance(data, str):
        _logs_common_func.warning(f'Data is not of type str, hashing may fail')
    hash_value = mmh3.hash64(data.encode())
    return hash_value[0]


_TIMEIT_STORE = {}
def measure_time(id: str = None, suffix:str = ''):
    """Logs the amount of time taken. Lets you use the function without having to write text bloating the code

    Args:
        id (str, optional): Pass the ID registered earlier. If None, it will record a new ID for timing and return an ID to be used for being called later. Defaults to None.
        suffix (str, optional): Any text that needs to be appended to the end of the log printed. Defaults to '', an empty string.

    Returns:
        str: Returns an ID for the timing that can be used later to measure time difference
    """
    global _TIMEIT_STORE
    if id:
        timenow = default_timer()
        timethen = _TIMEIT_STORE.get(id, timenow)
        timedelta = timenow - timethen
        _logs_common_func.info(f'Took {round(timedelta, 1)} seconds {suffix}')
        return None

    unique_id, thetime = get_uuid(), default_timer()
    _TIMEIT_STORE.update({unique_id: thetime})
    return unique_id


def parquet_normalise(data: pandas.DataFrame) -> pandas.DataFrame:
    """Used to remove unsupported format of data in a dataframe before writing to parquet. Handles list of multiple types of objects. These are not supported and all objects have to be of the same type. If such an instance is found, the columnm value is converted to string

    Errors handled:
    - cannot mix list and non-list, non-null values
    - ValueError: Can't infer object conversion type: 0
    """
    data = data.to_dict('records')
    for row in data:
        for k,v in row.items():
            if isinstance(v, list):
                if not all([isinstance(_, type(v[0])) for _ in v]):
                    row[k] = str(v)

    data = pandas.DataFrame(data)
    for col in data.columns:
            data_types = data[col].apply(lambda x: type(x)).unique()
            if list in data_types and len(data_types) > 1:
                data[col] = data[col].astype(str)
    return data