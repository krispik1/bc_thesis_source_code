from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from dataset_writer.dataset_schemas import Schema, ColumnSpecification


class H5Writer:
    """
    Class used to write data in a HDF5 file. Group of the file is equivalent to a table that follows given schema.
    Datasets inside that group represent the columns of the table given by the schema.
    """
    def __init__(
            self,
            schema: Schema,
            path: Path,
            chunks: int = 4096
    ) -> None:
        """
        Constructor for H5Writer.

        :param schema: Schema of a table that HDF5 will follow.
        :param path: File path for the HDF5 file.
        :param chunks: Chunks in which the HDF5 file will be written.
        """
        # Make a folder
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.chunks = chunks
        self.schema = schema

        # Open file in append mode and require a group that is the schema's name
        self.file = h5py.File(str(self.path), 'a')
        self.file_group = self.file.require_group(schema.name)

        # In group, create datasets for each column and save pointers to them in a dictionary
        self.group_datasets : Dict[str, h5py.Dataset] = {
            schema.column_names[i] : self._req_datasets(schema.column_names[i], schema.column_specifications[i])
            for i in range(schema.n_cols)}

    def _req_datasets(
            self,
            column_name: str,
            column_specification: ColumnSpecification
    ) -> h5py.Dataset:
        """
        Helper function for creating a dataset (by which we represent a column).

        If the dataset is already present, then return its pointer. If it is not present, then create it.

        :param column_name: Name of the dataset.
        :param column_specification: Specification of the dataset.
        :return: Pointer to the dataset.
        """
        if column_name in self.file_group:
            return self.file_group[column_name]

        chunks = (self.chunks, *column_specification.shape) if column_specification.shape else (self.chunks,)

        return self.file_group.create_dataset(
            column_name,
            shape=(0, *column_specification.shape),
            maxshape=(None, *column_specification.shape),
            dtype=column_specification.dtype,
            chunks=chunks
        )

    def add_batch(
            self,
            batch: Dict[str, List[float]]
    ) -> None:
        """
        Writes a batch of data representing number of episodes to the file.

        :param batch: Rows to add given by a dictionary of (column name, list of column values indexed by rows).
        """
        # Each list in dictionary must be of an equal length
        columns = list(batch.values())
        assert not any(len(columns[0]) != len(column) for column in columns)

        for i in range(self.schema.n_cols):
            # Get column content from batch and pointer to dataset corresponding to given column
            column = np.asarray(
                batch[self.schema.column_names[i]],
                dtype=self.schema.column_specifications[i].dtype,
            )
            ds = self.group_datasets[self.schema.column_names[i]]

            # Find sizes of new data, already written dataset and new size of written dataset
            b = int(column.shape[0])
            n = int(ds.shape[0])
            n1 = n + b

            # Resize the dataset and write the data
            ds.resize(n1, axis=0)

            ds[n:n1] = column

        self.file.flush()

    def close(
            self
    ) -> None:
        """
        Closes the file.
        """
        self.file.close()

    def get_size(
            self
    ) -> int:
        """
        Returns the current size of the dataset.
        
        :return: Size of the dataset.
        """
        
        return self.group_datasets[self.schema.column_names[0]].shape[0]
