import os, sys
from argparse import ArgumentParser

from pricePrediction.selectFromRawData.selectFromRawDataNonVirtualAll_base import RawDataSelector
from pricePrediction import config

class SelectFromRawDataNonVirtualAll(RawDataSelector):

    DESIRED_PARAMS_TO_ASK= ['full_dataset_fname', 'computed_datadir']
    def __init__(self, full_dataset_fname: str =
             os.path.join(config.RAW_DATA_DIRNAME, "mcule_purchasable_full_prices_210616_O9Jw1D_only_READILY.csv.gz"),
             computed_datadir: str = config.DATASET_DIRNAME):
        '''

        :param str full_dataset_fname: The fname for the whole Mcule csv file (csv.gz)
        :param str bb_fname: The fname for the building blocks csv file (csv.gz)
        :param str computed_datadir: The directory where the selected records will be saved
        '''
        super().__init__(full_dataset_fname, None, computed_datadir)

    def operate_data(self, data):

        # data = data[ ~ data.iloc[:,-1].isna() ]
        data = data[self.COLNAMES[:3]]
        data = data.rename(columns={self.COLNAMES[2]: "price"})
        del data["Mcule ID"]
        return data

if __name__ == "__main__":
    parser = ArgumentParser(description="extract from raw data train/test/val partitions")
    SelectFromRawDataNonVirtualAll.addParamsToArgParse(parser)
    args = parser.parse_args()
    SelectFromRawDataNonVirtualAll(**vars(args)).compute()

    '''
python -m pricePrediction.selectFromRawData.selectFromRawDataNonVirtualAll
    '''