import atexit
import logging
import os
import shutil

import dirsync
import joblib
import torch

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from torch.utils.data import DataLoader, Subset

from pricePrediction import config
from pricePrediction.ArgParser_base import ArgParseable
from pricePrediction.dataManager.dataset import Dataset_graphPrice
from pricePrediction.preprocessData.prepareDataMol2Price import DataBuilder
from pricePrediction.utils import EncodedDirNamesAndTemplates


class GraphPriceDatamodule(pl.LightningDataModule, ArgParseable):


    DESIRED_PARAMS_TO_ASK= ['encodedDir', 'batch_size', 'num_workers', 'use_lds', 'use_weights', 'force_compute_data',
                            'do_undersample', 'augment_labels', 'copy_data_to_wdir', 'debug']

    def __init__(self, encodedDir=config.ENCODED_DIR, deg_fname=None, batch_size: int = config.BATCH_SIZE,
                 num_workers: int = config.NUM_WORKERS_PER_GPU, random_seed: int = config.RANDOM_SEED,
                 use_log_for_target: bool = True, use_lds: bool = False,
                 use_weights: bool = False, force_compute_data: bool = False, do_undersample: bool = False,
                 augment_labels: bool =True, shuffle_training: bool =True, copy_data_to_wdir: str = None, debug =False):
        '''The DataManager that will create several DataLoader(s)

        :param str encodedDir: The directory where the dataset has been prepared as lmdb files.
        :param str deg_fname: The file containing the statistics about the nodes degree. By default is automatically searched within
                              encodedDir.
        :param int batch_size: Batch size.
        :param int num_workers: Number of workers as in "pytorch.DataLoader".
        :param int random_seed: Random seed.
        :param bool use_log_for_target: Apply log to the target (log(price)).
        :param bool use_lds: Use label density smoothing to reduce data imbalance.
        :param bool use_weights: Apply weights to reduce data imbalance.
        :param bool force_compute_data: Recompute all data before using the DataManager. Takes some time.
        :param bool do_undersample: Use undersampling to reduce data imbalance.
        :param bool augment_labels: Add random perturbation to training labels to reduce overfitting.
        :param bool shuffle_training: Shuffle training data.
        :param str copy_data_to_wdir: An optional working dir where data can be copied. Useful to avoid network disk usage.
        :param bool debug: Use only a small subset of the dataset
        '''

        seed_everything(random_seed)
        super().__init__()

        if encodedDir is None:
            encodedDir = config.ENCODED_DIR
        if deg_fname is None:
            names = EncodedDirNamesAndTemplates(encodedDir)
            deg_fname = names.DEGREES_FNAME

        self.deg_fname = deg_fname
        self.do_undersample = do_undersample
        self.use_weights = use_weights or use_lds
        self.use_lds = use_lds
        self.use_log_for_target = use_log_for_target
        self.random_seed = random_seed

        self.copy_data_to_wdir = copy_data_to_wdir

        self.encodedDir = encodedDir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_compute_data = force_compute_data
        self.augment_labels = augment_labels
        self.shuffle_training = shuffle_training

        self.debug = debug

        ###### Delayed variables
        self._dims = None
        self._train_n_instances = None
        self.setup_done = False
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def _getBucketsFname(self, training_phase):
        names = EncodedDirNamesAndTemplates(self.encodedDir)
        return names.BUCKETS_FNAME_TEMPLATE % training_phase

    @property
    def dims(self):
        if self._dims is None:
            metadata_fname = EncodedDirNamesAndTemplates(self.encodedDir).DATASET_METADATA_FNAME_TEMPLATE % "train"
            size_metadata_info = joblib.load(metadata_fname)
            self._dims = (size_metadata_info["nodes_n_features"], size_metadata_info["edges_n_features"])
            self._train_n_instances = size_metadata_info["total_size"]
        return self._dims

    def get_nodes_degree(self):
        if os.path.isfile( self.deg_fname ):
            return joblib.load(self.deg_fname)
        else:
            self.prepare_data()
            return self.get_nodes_degree()

    def prepare_data(self):
        # Compute if not already done
        if self.force_compute_data:
            dataBuilder = DataBuilder()
            dataBuilder.prepareDataset(encodedDir=self.encodedDir, datasetSplit="train")
            dataBuilder.prepareDataset(encodedDir=self.encodedDir, datasetSplit="test")
            dataBuilder.prepareDataset(encodedDir=self.encodedDir, datasetSplit="val")
        else:
            dataBuilder = DataBuilder()
            if not os.path.exists(os.path.join(self.encodedDir, "train")):
                dataBuilder.prepareDataset(encodedDir=self.encodedDir, datasetSplit="train")
            if not os.path.exists(os.path.join(self.encodedDir, "test")):
                dataBuilder.prepareDataset(encodedDir=self.encodedDir, datasetSplit="test")
            if not os.path.exists(os.path.join(self.encodedDir, "val")):
                dataBuilder.prepareDataset(encodedDir=self.encodedDir, datasetSplit="val")

        # if self.copy_data_to_wdir:
        #     tmpArgs = {}
        #     print(self.copy_data_to_wdir, os.path.isdir(self.copy_data_to_wdir))
        #     if not os.path.isdir(self.copy_data_to_wdir):
        #         tmpArgs["prefix"] = self.copy_data_to_wdir
        #         self._tmpdir = TemporaryDirectory(**tmpArgs)
        #         def cleanTmpdir():
        #             del self._tmpdir
        #         atexit.register( cleanTmpdir)
        #
        #         self.copy_data_to_wdir = self._tmpdir.name

        if self.copy_data_to_wdir:
            if not os.path.isdir(self.copy_data_to_wdir):
                os.makedirs(self.copy_data_to_wdir)
                _tmpdir = self.copy_data_to_wdir
                def cleanTmpdir():
                    shutil.rmtree( _tmpdir)
                atexit.register(cleanTmpdir)

            print("copying training data from %s to %s" % (self.encodedDir, self.copy_data_to_wdir))
            dirsync.sync(self.encodedDir, self.copy_data_to_wdir, 'sync', verbose=True, logger=logging.getLogger('dummy'))

    def setup(self, stage=None):
        if self.dataset_train is not None and self.dataset_val is not None and self.dataset_test is not None:
            return

        if self.copy_data_to_wdir is not None:
            encodedDir = self.copy_data_to_wdir
        else:
            encodedDir = self.encodedDir

        self.dataset_train = Dataset_graphPrice(encodedDir, datasetSplit="train",
                                                use_log_for_target=self.use_log_for_target,
                                                use_weights=self.use_weights, augment_labels = self.augment_labels,
                                                buckets_fname=self._getBucketsFname("train"),
                                                use_lds=self.use_lds, do_undersample=self.do_undersample)

        self.dataset_val = Dataset_graphPrice(encodedDir, datasetSplit="val",
                                              use_log_for_target=self.use_log_for_target,
                                              use_weights=self.use_weights, augment_labels = False,
                                              buckets_fname=self._getBucketsFname("val"),
                                              use_lds=self.use_lds, do_undersample=False)

        self.dataset_test = Dataset_graphPrice(encodedDir, datasetSplit="test",
                                                use_log_for_target=self.use_log_for_target,
                                                use_weights=self.use_weights, augment_labels = False,
                                                buckets_fname=self._getBucketsFname("test"),
                                                use_lds=self.use_lds, do_undersample=False)

        atexit.register( self.dataset_train.close )
        atexit.register( self.dataset_val.close )
        atexit.register( self.dataset_test.close )

    def _generic_dataloader(self, dataset, num_workers=None, **kwargs):
        if num_workers is None:
            num_workers = self.num_workers


        if num_workers>0 and "timeout" not in kwargs:
            kwargs["timeout"] = config.MULTIPROC_TIMEOUT

        if num_workers > 0 and "prefetch_factor" not in kwargs:
            kwargs["prefetch_factor"] = config.MULTIPROC_PREFETCH_FACTOR

        if num_workers > 0 and "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = config.PERSISTENT_WORKERS

        print("Using num_workers:", num_workers)
        #print(self.trainer.reload_dataloaders_every_epoch)

        from torch_geometric.data import Batch
        def from_list_of_examples_to_batch(examples):
            graphs, ys = zip(* examples)
            g_batch = Batch.from_data_list(graphs)
            ys = torch.stack(ys)
            return [g_batch, ys]
        if self.debug:
            dataset = Subset(dataset, range(21*self.batch_size))
        return DataLoader(dataset, num_workers=num_workers, batch_size=self.batch_size,
                          pin_memory=num_workers>0, collate_fn = from_list_of_examples_to_batch,
                          **kwargs)

    def train_dataloader(self, **kwargs):
        if self.trainer:
            print("Generating a train_dataloader at epoch: %d"% (  self.trainer.current_epoch  ))
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = False
        return self._generic_dataloader(self.dataset_train,  shuffle=True, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._generic_dataloader(self.dataset_val, shuffle=False, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._generic_dataloader(self.dataset_test, shuffle=False, **kwargs)

    def predict_dataloader(self, dataset=None):

        if dataset == "train":
            raise NotImplementedError()
        elif dataset == "test":
            raise NotImplementedError()
        elif dataset == "val":
            raise NotImplementedError()
        if dataset is not None:  # TODO: check type
            raise NotImplementedError()
        else:
            raise NotImplementedError()



if __name__ == "__main__":

    from pricePrediction.ArgParser_base import ArgParseable, MyArgParser

    parser = MyArgParser(prog=None, usage=None, description=None, )

    group = parser.add_argument_group(title="data")
    GraphPriceDatamodule.addParamsToArgParse(group)
    cmd_args = parser.parse_args()
    dm = GraphPriceDatamodule(**cmd_args.get("data", {}))
    dm.setup()
    allPrices = []
    for batchG, batchY in dm.test_dataloader():
        allPrices.extend( batchY.tolist() )

    import matplotlib.pyplot as plt
    plt.hist(allPrices, bins=200)
    plt.show()

    '''
python -m pricePrediction.dataManager.dataManager
    '''
