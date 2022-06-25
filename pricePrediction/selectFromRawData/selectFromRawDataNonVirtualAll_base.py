import os
import dask.dataframe as dd
import pandas as pd

from pricePrediction import config
from pricePrediction.ArgParser_base import ArgParseable
from pricePrediction.config import TEST_SIZE


class RawDataSelector(ArgParseable):
    COLNAMES = "Mcule ID,SMILES,price 1 (USD),amount 1 (mg),delivery time 1 (w.days),available amount 1 (mg)".split(",")

    DESIRED_PARAMS_TO_ASK= ['full_dataset_fname', 'bb_fname', 'computed_datadir', 'test_size', 'nrows']
    def __init__(self, full_dataset_fname: str, bb_fname: str= config.BUILDING_BLOCKS_FNAME,
                 computed_datadir: str= config.DATASET_DIRNAME, test_size:int=TEST_SIZE, nrows:int=None):
        '''

        :param str full_dataset_fname: The fname for the whole Mcule csv file (csv.gz)
        :param str bb_fname: The fname for the building blocks csv file (csv.gz)
        :param str computed_datadir: The directory where the selected records will be saved
        :param int test_size: The number of entries to include in the test set
        :param int nrows: The number of rows to process in the full_dataset_fname. If None, process all rows
        '''
        self.full_dataset_fname = full_dataset_fname
        self.bb_fname = bb_fname
        self.computed_datadir = computed_datadir
        self.test_size = test_size
        self.nrows = nrows

    def operate_data(self, data):
        raise NotImplementedError()

    def compute(self):
        if self.full_dataset_fname.endswith(".gz"):
            data = pd.read_csv(self.full_dataset_fname, usecols=self.COLNAMES, nrows=self.nrows)
            data = dd.from_pandas(data, npartitions=30)
        else:
            data = dd.read_csv(self.full_dataset_fname, usecols=self.COLNAMES,blocksize=None)
            if self.nrows is not None:
                data = data.head(n=self.nrows, compute=False)

        data = self.operate_data(data)
        self.saveDataSplits(data)


    def saveDataSplits(self, data, computed_datadir=None):
        if computed_datadir is None:
            computed_datadir = self.computed_datadir
        print("Saving results to %s"%computed_datadir)
        if isinstance(data, dd.DataFrame):
            n_elems = data.shape[0].compute()
        print("Total size: ", n_elems)
        test_frac = self.test_size/n_elems
        data_train, data_test = data.random_split([1-test_frac, test_frac])

        del data

        data_test = data_test.repartition(npartitions= int(n_elems*test_frac//self.test_size)+1)
        data_test.to_csv(os.path.join(computed_datadir, "mcule_full_test.csv_split_*.csv"), index=False)
        del data_test

        n_elems = data_train.shape[0].compute()
        val_frac = self.test_size/n_elems
        data_train, data_val = data_train.random_split([1-val_frac, val_frac])
        data_val= data_val.repartition(npartitions= int(n_elems*val_frac//self.test_size)+1)

        data_val.to_csv(os.path.join(computed_datadir, "mcule_full_val.csv_split_*.csv"), index=False)
        del data_val

        data_train= data_train.repartition(npartitions= int(n_elems*(1-val_frac)//self.test_size)+1)
        data_train.to_csv(os.path.join(computed_datadir, "mcule_full_train.csv_split_*.csv"), index=False)
        with open(os.path.join(computed_datadir,"params.txt"), "w") as f:
            f.write(str(type(self))+"\n" )
            f.write(str(self.full_dataset_fname)+"\n" )
            f.write(str(self.bb_fname)+"\n" )
            f.write(str(self.computed_datadir))