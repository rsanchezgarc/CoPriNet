import os, sys
import dask.dataframe as dd
from argparse import ArgumentParser

from pricePrediction.config import TEST_SIZE
from pricePrediction.selectFromRawData.selectFromRawDataNonVirtualAll_base import RawDataSelector
import pandas as pd
from pricePrediction import config

class SelectFromRawDataOnlyVirtualAll(RawDataSelector):
    DESIRED_PARAMS_TO_ASK= ['full_dataset_fname', 'bb_fname', 'computed_datadir',]
    def __init__(self, full_dataset_fname: str =
             os.path.join(config.RAW_DATA_DIRNAME, "mcule_purchasable_full_prices_210319_kLp4Zt_march.csv.gz"),
             bb_fname: str = config.BUILDING_BLOCKS_FNAME,
             computed_datadir: str = config.DATASET_DIRNAME,
             max_size:int=10000000,
             test_size:int=TEST_SIZE, nrows:int=None):
        '''
        :param str full_dataset_fname: The fname for the whole Mcule csv file (csv.gz)
        :param str bb_fname: The fname for the building blocks csv file (csv.gz)
        :param str computed_datadir: The directory where the selected records will be saved
        :param int test_size: The number of entries to include in the test set
        :param int nrows: The number of rows to process in the full_dataset_fname. If None, process all rows
        '''
        super().__init__(full_dataset_fname, bb_fname, computed_datadir,test_size=test_size,nrows=nrows)
        self.max_size = max_size

    def read_csv(self, csv_name):
        data = dd.read_csv(csv_name, usecols=self.COLNAMES,blocksize=None)
        data = data[ data[self.COLNAMES[-1]].isna() ] #To select compounds that are not in stock
        data = data[self.COLNAMES[:3]]
        data = data.rename(columns={self.COLNAMES[2]: "price"})
        return data

    def operate_data(self, data):
        data = self.read_csv(self.full_dataset_fname)
        size = data.shape[0].compute()
        bb = pd.read_csv(self.bb_fname, sep="\t", header=None)
        bb.columns = ["SMILES", "Mcule ID"]
        del bb["SMILES"]
        data = data[ (~data["Mcule ID"].isin(bb["Mcule ID"])) ]
        size = data.shape[0].compute()
        print("Total size: ",size)
        sampling_size = min(size, self.max_size)
        data = data.sample(frac= sampling_size/size)
        print("Sampled size: ",sampling_size)
        data = data[["SMILES", "price"]]
        del bb
        return data

if __name__ == "__main__":
    parser = ArgumentParser(description="extract from raw data train/test/val partitions")
    SelectFromRawDataOnlyVirtualAll.addParamsToArgParse(parser)
    args = parser.parse_args()
    SelectFromRawDataOnlyVirtualAll(**vars(args)).compute()
