from pathlib import Path
from typing import Dict, Any

from dataset_writer.buffer import RowBuffer
from dataset_writer.dataset_schemas import Schema
from dataset_writer.h5_writer import H5Writer


class WriterManager:
    """
    Class that combines a HDF5 writer and a buffer following the same schema and methods to utilize for permanent
    data storage during data generation.
    """
    def __init__(
            self,
            schema: Schema,
            path: Path,
            buffer_size: int,
            chunks: int = 4096
    ) -> None:
        """
        Constructor for WriterManager.

        :param schema: Schema of the table the buffer and writer will follow.
        :param path: Path to the HDF5 file.
        :param buffer_size: Size of the buffer.
        :param chunks: Chunk sizes in the HDF5 file.
        """
        self.schema = schema
        self.path = path
        self.chunks = chunks

        self.writer = H5Writer(self.schema, path, chunks)
        self.buffer = RowBuffer(self.schema, buffer_size)

    def save_data(
            self,
            row: Dict[str, Any]
    ) -> None:
        """
        Adds data to the buffer. If the buffer is full, the data is saved to the HDF5 file.

        :param row: Dictionary of (column name, row value).
        """
        if self.buffer.add_row(row):
            self.writer.add_batch(self.buffer.get_batch())

    def close(
            self
    ) -> None:
        """
        Closes the HDF5 writer but stores contents of the buffer that have not been written yet.
        """
        batch = self.buffer.get_batch()
        if batch and len(next(iter(batch.values()))) > 0:
            self.writer.add_batch(batch)
        self.writer.close()

    def change_h5_path(
            self,
            path: Path
    ) -> None:
        """
        Changes the path of the HDF5 file by closing the old one and creating a new H5Writer.

        :param path: New path of the HDF5 file.
        """
        self.path = path

        self.close()
        self.writer = H5Writer(self.schema, path, self.chunks)

    def get_size(
            self
    ) -> int:
        """
        Returns the current size of the dataset.

        :return: Size of the dataset.
        """
        
        return self.writer.get_size()
