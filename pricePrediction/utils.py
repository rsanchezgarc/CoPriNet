import os
import shutil
import numpy as np

from pricePrediction import config

def tryMakedir(dirname, remove=False):
    if  os.path.isdir(dirname):
        if remove:
            # print("removing %s"%dirname)
            shutil.rmtree(dirname)
            os.mkdir(dirname)
    else:
        os.mkdir(dirname)


def getBucketRanges(x0=1,x1=11, n_buckets=config.NUM_BUCKETS):
    return np.linspace(x0, x1, num=n_buckets)


def search_buckedId(y, bucket_ranges):
    n_buckets = len(bucket_ranges)
    return min(n_buckets - 1, np.searchsorted(bucket_ranges, y))


class EncodedDirNamesAndTemplates(object):
    def __init__(self, encodedDir):
        self.DEGREES_FNAME = os.path.join(encodedDir, "graphs_degrees.joblib.pckl")
        self.BUCKETS_FNAME_TEMPLATE = os.path.join(encodedDir, "n_per_bucket.%s.joblib.pckl")
        self.DIR_RAW_DATA_SELECTED= os.path.join(encodedDir, "selected_smiles_price")
        self.SELECTED_DATAPOINTS_TEMPLATE = os.path.join(self.DIR_RAW_DATA_SELECTED, "selected_smiles_price.%s.%s.csv.gz")
        self.DATASET_METADATA_FNAME_TEMPLATE = os.path.join(encodedDir, "%s_metadata.joblib.pckl")