import os
import joblib
import lmdb
import numpy as np

from torch.utils.data import Dataset

from pricePrediction.preprocessData.serializeDatapoints import getExampleId, deserializeExample
from pricePrediction.utils import search_buckedId, EncodedDirNamesAndTemplates


class Dataset_graphPrice(Dataset):
    def __init__(self, encodedDir, datasetSplit, use_log_for_target=True, use_weights=False, buckets_fname=None,
                 use_lds=False, do_undersample=False, augment_labels=False):

        self.use_log_for_target = use_log_for_target
        self.use_weights = use_weights
        self.buckets_fname = buckets_fname
        self.use_lds = use_lds
        self.do_undersample = do_undersample
        self.augment_labels = augment_labels
        assert (self.buckets_fname is not None and self.use_weights) or not self.use_weights
        if self.do_undersample: raise NotImplementedError()

        dataNames = EncodedDirNamesAndTemplates(encodedDir)
        size_metadata_info = joblib.load( dataNames.DATASET_METADATA_FNAME_TEMPLATE % datasetSplit)

        self.total_size = size_metadata_info["total_size"]
        self.size_per_file = size_metadata_info["sizes_list"]
        self.db_fnames = [ os.path.join( encodedDir, datasetSplit, os.path.basename(fname))
                                        for fname in size_metadata_info["fnames_list"]]
        self.cum_size_at_file = np.cumsum(np.array(size_metadata_info["sizes_list"], dtype=int))
        self.last_cum_elemIdx_at_file = self.cum_size_at_file - 1

        self.fileIdx_to_fileBasename = [os.path.basename(fname) for fname in self.db_fnames]

        #Delayed load
        self.db_managers= [None]*len(self.size_per_file)

        self._bucket_ranges = None
        self._n_per_bucket = None

    def _idx_to_file_and_num(self, idx):
        fileIdx = search_buckedId(idx, self.last_cum_elemIdx_at_file)
        idx = idx - (self.cum_size_at_file[fileIdx - 1] if fileIdx - 1 >= 0 else 0)
        return fileIdx, idx

    def _init_db(self, fileIdx):
        env = lmdb.open(self.db_fnames[fileIdx],
            readonly=True, lock=False,
            readahead=False, meminit=False)
        txn = env.begin()
        self.db_managers[fileIdx] = (env, txn)
        return env, txn

    @property
    def bucket_ranges(self):
        if self._bucket_ranges is None:
            self._getBuckets()
        return self._bucket_ranges

    @property
    def n_per_bucket(self):
        if self._n_per_bucket is None:
            self._getBuckets()
        return self._n_per_bucket

    def n_batches(self, batch_size):
        return len(self)//batch_size + int(bool(len(self)%batch_size))

    def _getBuckets(self):
        buckets = joblib.load(self.buckets_fname)
        bucket_ranges, n_per_bucket = buckets["bucket_ranges"], buckets["n_per_bucket"]

        if self.use_lds:
            print("using lds")
            from scipy.ndimage import gaussian_filter1d
            n_per_bucket = gaussian_filter1d(n_per_bucket, sigma=2)
        self._bucket_ranges = bucket_ranges
        self._n_per_bucket = n_per_bucket

    def compute_weight(self, label):
        if not self.use_log_for_target:
            label = np.exp(label)
        bucket_num = search_buckedId(label, self.bucket_ranges)
        w = self.total_size / (1 + self.n_per_bucket[bucket_num])
        return w

    def prepare_undersample(self, maximum_percentile=80):
        maximum_per_bucket = np.percentile(self.n_per_bucket, maximum_percentile)
        prob_per_bucket = maximum_per_bucket / (1e-10 + self.n_per_bucket)
        prob_per_bucket = np.where(prob_per_bucket < 1, prob_per_bucket, 1)
        print("maximum_per_bucket while sampling: %d" % maximum_per_bucket)
        new_len = sum([min(maximum_per_bucket, n_elems) for n_elems in self.n_per_bucket])

        def predicate(graph_y):
            y = graph_y[1]
            bucket_num = search_buckedId(y, self.bucket_ranges)
            return np.random.rand() < prob_per_bucket[bucket_num]

        # dataset = dataset.select(predicate)
        raise NotImplementedError() #This code is obsolete and requires refactoring

    def __getitem__(self, index):
        if index >= self.total_size:
            raise IndexError()
        fileIdx, pointIdx = self._idx_to_file_and_num(index)
        db_manager = self.db_managers[fileIdx]
        if db_manager is None:
            txn= self._init_db(fileIdx)[-1]
        else:
            txn = db_manager[-1]

        dataBytes = txn.get(getExampleId(self.fileIdx_to_fileBasename[fileIdx], pointIdx))
        if dataBytes is None:
            msg = "Error:\n%s %s   %s   %s"%(index, fileIdx, pointIdx, getExampleId(self.fileIdx_to_fileBasename[fileIdx], pointIdx))
            raise Exception(msg)
        graph, label = deserializeExample(dataBytes)
        if self.use_log_for_target:
            label = np.log(label)

        if self.augment_labels:
            label = label * (1+0.05*np.random.rand()-0.025)

        if self.use_weights:
            graph.w = self.compute_weight(label)
        else:
            graph.w = 1
        graph.y = label
        return graph, label

    def __len__(self):
        return self.total_size

    def close(self):
        for db_handler in self.db_managers:
            if db_handler:
                env, txn = db_handler
                env.close()