import os
import dask.dataframe as dd
from pricePrediction import config
from pricePrediction.ArgParser_base import ArgParseable
from pricePrediction.config import TEST_SIZE


class RawDataSelector(ArgParseable):
    COLNAMES = "Mcule ID,SMILES,price 1 (USD),amount 1 (mg),delivery time 1 (w.days),available amount 1 (mg)".split(",")

    DESIRED_PARAMS_TO_ASK= ['full_dataset_fname', 'bb_fname', 'computed_datadir']
    def __init__(self, full_dataset_fname: str, bb_fname: str= config.BUILDING_BLOCKS_FNAME,
                 computed_datadir: str= config.DATASET_DIRNAME):
        '''

        :param str full_dataset_fname: The fname for the whole Mcule csv file (csv.gz)
        :param str bb_fname: The fname for the building blocks csv file (csv.gz)
        :param str computed_datadir: The directory where the selected records will be saved
        '''
        self.full_dataset_fname = full_dataset_fname
        self.bb_fname = bb_fname
        self.computed_datadir = computed_datadir


    def operate_data(self, data):
        raise NotImplementedError()

    def compute(self):
        data = dd.read_csv(self.full_dataset_fname, usecols=self.COLNAMES,blocksize=None)

        data = self.operate_data(data)
        self.saveDataSplits(data)


    def saveDataSplits(self, data, computed_datadir=None):
        if computed_datadir is None:
            computed_datadir = self.computed_datadir
        print("Saving results to %s"%computed_datadir)
        n_elems = data.shape[0].compute()
        print("Total size: ", n_elems)
        test_frac = TEST_SIZE/n_elems
        data_train, data_test = data.random_split([1-test_frac, test_frac])

        del data

        data_test = data_test.repartition(npartitions= int(n_elems*test_frac//TEST_SIZE)+1)
        data_test.to_csv(os.path.join(computed_datadir, "mcule_full_test.csv_split_*.csv"), index=False)
        del data_test

        n_elems = data_train.shape[0].compute()
        val_frac = TEST_SIZE/n_elems
        data_train, data_val = data_train.random_split([1-val_frac, val_frac])
        data_val= data_val.repartition(npartitions= int(n_elems*val_frac//TEST_SIZE)+1)

        data_val.to_csv(os.path.join(computed_datadir, "mcule_full_val.csv_split_*.csv"), index=False)
        del data_val

        data_train= data_train.repartition(npartitions= int(n_elems*(1-val_frac)//TEST_SIZE)+1)
        data_train.to_csv(os.path.join(computed_datadir, "mcule_full_train.csv_split_*.csv"), index=False)
        with open(os.path.join(computed_datadir,"params.txt"), "w") as f:
            f.write(str(type(self))+"\n" )
            f.write(str(self.full_dataset_fname)+"\n" )
            f.write(str(self.bb_fname)+"\n" )
            f.write(str(self.computed_datadir))