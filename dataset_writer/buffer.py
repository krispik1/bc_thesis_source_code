from typing import Dict, Any, List

from dataset_writer.dataset_schemas import Schema

class RowBuffer:
    """
    Class representing a buffer that stores rows of tables for batch writing into tables to reduce I/O operations of
    small size of data. The data is stored by column but the whole buffer represents rows of a table.
    """

    def __init__(
            self,
            schema: Schema,
            buffer_size: int
    ) -> None:
        """
        Constructor for RowBuffer class. Buffer is constructed for one table that is given by its schema.

        :param schema: Schema used to structure the buffer.
        :param buffer_size: Size of the buffer.
        """
        self.schema = schema
        self.buffer_size = buffer_size

        self.column_buffers: Dict[str, List[float]] = {k: [] for k in schema.column_names}

    def _is_full(
            self
    ) -> bool:
        """
        Checks if the buffer is full.

        :return: True only if the buffer is full.
        """
        return len(list(self.column_buffers.values())[0]) == self.buffer_size

    def _empty_buffer(
            self,
    ) -> None:
        """
        Emptys the buffer.
        """
        self.column_buffers = {k: [] for k in self.schema.column_names}

    def get_batch(
            self
    ) -> Dict[str, List[float]]:
        """
        Gets a batch of rows stored in buffer and empties the buffer for new data.

        :return: Dictionary of (column name, list of column values indexed by rows).
        """
        batch = self.column_buffers
        self._empty_buffer()
        return batch

    def add_row(
            self,
            row: Dict[str, Any]
    ) -> bool:
        """
        Adds a row to the buffer.

        :param row: Dictionary of (column name, row value).
        :return: True only if the buffer is full after adding row.
        """
        for col in self.schema.column_names:
            self.column_buffers[col].append(row[col])

        return self._is_full()
