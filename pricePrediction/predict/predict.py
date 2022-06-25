import argparse
import os, sys
from itertools import islice
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from more_itertools import chunked
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import pytorch_lightning as pl
from pricePrediction.nets.netsGraph import PricePredictorModule
from pricePrediction.config import NUM_WORKERS_PER_GPU, BATCH_SIZE, DEFAULT_MODEL, USE_FEATURES_NET, \
    BUFFER_N_BATCHES_FOR_PRED


class GraphPricePredictor():
    name = "price_GNN"

    def __init__(self, model_path=DEFAULT_MODEL, n_gpus = 1, n_cpus= NUM_WORKERS_PER_GPU, batch_size:int=BATCH_SIZE,
                 **kwargs):
        self.model_path = model_path
        self.n_gpus = n_gpus
        self.n_cpus = n_cpus
        self.batch_size = batch_size
        self.trainer = pl.Trainer(gpus=self.n_gpus, logger=False)
        self.model = PricePredictorModule.load_from_checkpoint(self.model_path, batch_size=self.batch_size)

        if USE_FEATURES_NET:
            from pricePrediction.preprocessData.smilesToDescriptors import smiles_to_graph

        else:
            from pricePrediction.preprocessData.smilesToGraph import smiles_to_graph

        self.smiles_to_graph = smiles_to_graph

    def prepare_smi(self, idx_smi):
        idx, smi = idx_smi
        graph = self.smiles_to_graph(smi)
        if graph is None:
            return None
        graph.input_idx = idx
        return graph

    def yieldPredictions(self, smiles_generator, buffer_n_batches=BUFFER_N_BATCHES_FOR_PRED):
        buffer_size = buffer_n_batches * self.batch_size
        preds_iter = map(lambda x: self.predictListOfSmiles(x), tqdm(chunked(smiles_generator, buffer_size)))
        for preds_batch in preds_iter:
            for pred in preds_batch:
                yield pred

    def predictListOfSmiles(self, smiles_list):
        smiles_list = list(smiles_list)
        graphs_list = list(filter(None.__ne__, map(self.prepare_smi,  enumerate(smiles_list) )))
        graphs_fn = lambda : graphs_list
        dataset = MyIterableDataset(graphs_fn, self.n_cpus)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=Batch.from_data_list,
                                num_workers=self.n_cpus)

        preds = self.trainer.predict(self.model, dataloader)
        n_smiles = len(smiles_list)
        all_preds = np.nan * np.ones(n_smiles)
        for i, batch in enumerate(dataloader):
            batch_preds = preds[i].to("cpu").numpy()
            idxs = batch.input_idx.to("cpu").numpy().astype(np.int64).tolist()
            all_preds[idxs] = batch_preds
        return all_preds


class MyIterableDataset(IterableDataset):
    def __init__(self, generator_fun, num_workers):
        super().__init__()
        self.generator_fun = generator_fun
        self.num_workers = num_workers

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            uid = torch.utils.data.get_worker_info().id
            return islice( self.generator_fun(), uid, None, self.num_workers)
        else:
            return iter(self.generator_fun())


def main():
    parser = argparse.ArgumentParser(prog="CoPriNet")
    parser.add_argument("input_csv_file", type=str, help="The input csv filename containing a smiles column")
    parser.add_argument("-o", "--output_file", type=str, required=False, default=None, help="The output filename that will be "
                            "identical to the input_csv_file but with one additional column for the f1")
    parser.add_argument("--smiles_colname", type=str, required=False, default="SMILES", help="The colname for SMILES "
                                                                             "in the input file.  Default: %(default)s")
    parser.add_argument("--model_path", type=str, required=False, default=DEFAULT_MODEL,
                        help="The CoPriNet model checkpoing path. Default: %(default)s")

    parser.add_argument("--n_cpus", type=int, required=False, default=NUM_WORKERS_PER_GPU,
                        help="The number of cpu workers. Default: %(default)s")

    parser.add_argument("--batch_size", type=int, required=False, default=BATCH_SIZE,
                        help="Batch size. Default: %(default)s")

    parser.add_argument("--convert_to_g", action="store_true", help="Convert the f1 from $/mmol to $/g")

    coprinet_colname = "CoPriNet"

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_file)
    nans = df[args.smiles_colname].isna()
    df = df[~nans]
    smiles_list = df[args.smiles_colname]
    predictor = GraphPricePredictor( **vars(args))
    preds = predictor.yieldPredictions(smiles_list)
    if args.convert_to_g:
        from rdkit import Chem
        def convert_pred(smi, pred):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return np.nan
            mw = Chem.Descriptors.ExactMolWt(mol)
            price = np.log(  np.exp(pred)*1000/mw)
            return price

        preds = Parallel(n_jobs=args.n_cpus)(delayed(convert_pred)(smi, pred) for smi, pred in zip(smiles_list, preds))
    if args.output_file is None:
        for smi, pred in zip(smiles_list, preds):
            print("%s\t%.4f" % (smi, pred))
    else:
        df[coprinet_colname] = list(preds)
        df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()