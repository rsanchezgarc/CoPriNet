import gzip
import os
import re
import sys
import time
from functools import reduce
from itertools import chain
from multiprocessing import cpu_count

import lmdb
import psutil
import joblib
import torch
from joblib import Parallel, delayed
import numpy as np
from pricePrediction import config
from pricePrediction.config import USE_MMOL_INSTEAD_GRAM
from pricePrediction.preprocessData.serializeDatapoints import getExampleId, serializeExample
from pricePrediction.utils import tryMakedir, getBucketRanges, search_buckedId, EncodedDirNamesAndTemplates
from .smilesToGraph import compute_nodes_degree, fromPerGramToPerMMolPrice

if config.USE_FEATURES_NET:
    from .smilesToDescriptors import smiles_to_graph

else:
    from .smilesToGraph import smiles_to_graph

PER_WORKER_MEMORY_GB = 2

class DataBuilder():

    def __init__(self, n_cpus= config.N_CPUS):
        if n_cpus is None:
            mem_gib = psutil.virtual_memory().available / (1024. ** 3)
            n_cpus = int(max(1, min(cpu_count(), mem_gib // PER_WORKER_MEMORY_GB)))

        self.n_cpus = n_cpus

    def processOneFileOfSmiles(self, encodedDir, fileNum, datasetSplit, fname, nrows=None):
        print("processing %s"%fname)
        if fname.endswith(".csv"):
            open_fun = open
            decode_line = lambda line: line
        elif fname.endswith(".csv.gz"):
            open_fun = gzip.open
            decode_line = lambda line : line.decode('utf-8')
        else:
            raise ValueError("Bad file format")

        cols = ['SMILES','price']

        names = EncodedDirNamesAndTemplates(encodedDir)
        outFname_base = datasetSplit + "_" + str(fileNum) + "_lmdb"
        outFname = os.path.join(encodedDir, datasetSplit,outFname_base)
        env = lmdb.open(outFname,  map_size=10737418240)
        one_example_graph = smiles_to_graph("CCOCCC")
        degs = compute_nodes_degree(None)
        min_in_x = torch.ones(one_example_graph.x.shape[-1])*float('inf')
        max_in_x = -torch.ones(one_example_graph.x.shape[-1])*float('inf')

        num_examples = 0
        bucket_ranges = getBucketRanges()
        n_per_bucket = np.zeros(len(bucket_ranges), dtype= np.int64)

        with open_fun(fname) as f_in:
            header = decode_line(f_in.readline()).strip().split(",")
            try:
                smi_index, price_index = [ header.index(col) for col in cols]
            except ValueError:
                smi_index, price_index = 0,1
            with env.begin(write=True) as sink, \
                    gzip.open(names.SELECTED_DATAPOINTS_TEMPLATE % (datasetSplit, fileNum), "wt") as f_out:
                f_out.write("SMILES,price\n")
                cur_time = time.time()
                for i, line in enumerate(f_in):
                    lineArray = decode_line(line).strip().split(",")
                    smi, price = lineArray[smi_index], lineArray[price_index]
                    price = float(price)
                    graph = smiles_to_graph(smi)
                    if graph is None:
                        continue
                    #Save the original smiles-price
                    f_out.write("%s,%s\n"%(smi, price))

                    # Use the per mmol price
                    if USE_MMOL_INSTEAD_GRAM:
                        price = fromPerGramToPerMMolPrice(price, smi)
                    bucketId = search_buckedId( np.log(price), bucket_ranges)
                    n_per_bucket[bucketId] += 1

                    degs += compute_nodes_degree([graph])
                    min_in_x = torch.stack([min_in_x, torch.min(graph.x, 0)[0]]).min(0)[0]
                    max_in_x = torch.stack([max_in_x, torch.max(graph.x, 0)[0]]).max(0)[0]

                    fileId= getExampleId(outFname_base, num_examples)
                    sink.put(fileId, serializeExample(price, graph))
                    num_examples += 1
                    if nrows is not None and num_examples > nrows:
                        break
                    if i % 10000 == 0 and fileNum % self.n_cpus == 0:
                        new_time = time.time()
                        print("Current iteration: %d # task: %d  (%.2f s)       " % (i, fileNum, new_time - cur_time), end="\r")
                        cur_time = new_time
        if fileNum % self.n_cpus == 0:
            print()
        return ((outFname, degs, min_in_x, max_in_x, num_examples, n_per_bucket),)


    def getNFeatures(self):
        one_graph = smiles_to_graph("CCCCCCO")
        # print(one_graph)
        return dict(nodes_n_features=one_graph["x"].shape[-1], edges_n_features=one_graph["edge_attr"].shape[-1])

    def prepareDataset(self, inputDir=config.DATASET_DIRNAME, encodedDir=config.ENCODED_DIR, datasetSplit="train",
                       nrows=None, **kwargs):

        assert datasetSplit in ["train", "val", "test"]
        print("Computing %s dataset" % datasetSplit)
        print("Using %d workers for data preparation"%self.n_cpus)

        # os.environ["OMP_NUM_THREADS"] = "1"
        # os.environ["MKL_NUM_THREADS"] = "1"

        names = EncodedDirNamesAndTemplates(encodedDir)

        tryMakedir(encodedDir, remove=False)
        tryMakedir(os.path.join(encodedDir, datasetSplit))
        tryMakedir(names.DIR_RAW_DATA_SELECTED)


        fnames = [os.path.join(inputDir, fname) for fname in os.listdir(inputDir) if
                  re.match(config.RAW_DATA_FILE_SUFFIX, fname) and datasetSplit in fname]

        assert len(fnames) > 0

        results = Parallel(n_jobs=self.n_cpus, batch_size=1,
                           verbose=10)(delayed(self.processOneFileOfSmiles)(encodedDir, i, datasetSplit, fname, nrows=nrows)
                                                                        for i, fname in enumerate(fnames))
        results = chain.from_iterable(results)
        results = list(results)
        # print( results )
        fnames_list, degrees, min_in_x, max_in_x, sizes_lis, n_per_bucket = zip(*results)

        degrees = reduce(lambda prev, x: prev + x, degrees).numpy().tolist()
        min_in_x = reduce(lambda prev, x: torch.stack([prev, x]).min(0)[0], min_in_x).numpy().tolist()
        max_in_x = reduce(lambda prev, x: torch.stack([prev, x]).max(0)[0], max_in_x).numpy().tolist()

        n_per_bucket = reduce(lambda prev, x: prev + x, n_per_bucket)
        metadata_dict = {"name":datasetSplit, "fnames_list":fnames_list, "sizes_list": sizes_lis,
                      "total_size": sum(sizes_lis), "feature_stats":{"min":min_in_x, "max":max_in_x}}
        metadata_dict.update(self.getNFeatures())
        # print(metadata_dict)
        joblib.dump(metadata_dict,
                    names.DATASET_METADATA_FNAME_TEMPLATE % datasetSplit)
        joblib.dump(degrees, names.DEGREES_FNAME)

        # print(n_per_bucket)
        joblib.dump({"bucket_ranges": getBucketRanges(), "n_per_bucket":n_per_bucket}, names.BUCKETS_FNAME_TEMPLATE % datasetSplit)
        print("Dataset %s computed" % datasetSplit)
        return fnames_list



if __name__ == "__main__":
    print( " ".join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputDir", type=str, default=config.DATASET_DIRNAME, help="Directory where smiles-price pairs are located")
    parser.add_argument("-o", "--encodedDir", type=str, default=config.ENCODED_DIR)
    parser.add_argument("-n", "--ncpus", type=int, default=config.N_CPUS)
    parser.add_argument("--nrows", type=int, default=None, help="The number of rows to process in each file")

    args = vars( parser.parse_args())
    config.N_CPUS = args.get("ncpus", config.N_CPUS)

    dataBuilder = DataBuilder(n_cpus=config.N_CPUS)
    dataBuilder.prepareDataset(datasetSplit="train", **args)
    dataBuilder.prepareDataset(datasetSplit="val", **args)
    dataBuilder.prepareDataset(datasetSplit="test", **args)

'''
python -m  pricePrediction.preprocessData.prepareDataMol2Price


'''