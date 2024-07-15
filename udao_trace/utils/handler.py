import json
import os
import pickle
import traceback
from typing import Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class JsonHandler:
    @staticmethod
    def load_json(file: str) -> Dict:
        assert os.path.exists(file), FileNotFoundError(file)
        with open(file) as f:
            try:
                return json.load(f)
            except Exception as e:
                raise Exception(f"failed to load {file} with error: {e}")

    @staticmethod
    def load_json_from_str(s: str) -> Dict:
        return json.loads(s)

    @staticmethod
    def dump_to_string(obj: Dict, indent: Optional[int] = None) -> str:
        return json.dumps(obj, indent=indent)

    @staticmethod
    def dump_to_file(obj: Dict, file: str, indent: Optional[int] = None) -> None:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w") as f:
            json.dump(obj, f, indent=indent)


class PickleHandler(object):
    @staticmethod
    def save(obj: object, header: str, file_name: str, overwrite: bool = False) -> None:
        path = f"{header}/{file_name}"
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} already exists")
        os.makedirs(header, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(header: str, file_name: str) -> object:
        path = f"{header}/{file_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            return pickle.load(f)


class FileHandler:
    @staticmethod
    def create_script(header: str, file: str, content: str) -> None:
        os.makedirs(header, exist_ok=True)
        with open(f"{header}/{file}", "w") as f:
            f.write(content)


class ParquetHandler(object):
    @staticmethod
    def save(
        df: pd.DataFrame, header: str, file_name: str, overwrite: bool = False
    ) -> None:
        path = f"{header}/{file_name}"
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} already exists")
        os.makedirs(header, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)

    @staticmethod
    def load(header: str, file_name: str) -> pd.DataFrame:
        path = f"{header}/{file_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pq.read_table(path).to_pandas()


def error_handler(e: BaseException) -> None:
    print("An error occurred:")

    # Print the exception type
    print(f"Exception Type: {type(e).__name__}")

    # Print the exception message
    print(f"Exception Message: {str(e)}")

    # Print the traceback information
    print("Traceback:")
    traceback.print_exc()
