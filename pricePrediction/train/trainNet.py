import json
import sys, os

# import torch
# torch.multiprocessing.set_start_method('spawn')
# torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

from pytorch_lightning.plugins import DDPPlugin

from pricePrediction.ArgParser_base import MyArgParser
from pricePrediction.dataManager.dataManager import GraphPriceDatamodule
from pricePrediction.evaluation.evaluation import PlotFigsTensorboardCallback, InferencePlotter
from pricePrediction.nets.netsGraph import PricePredictorModule

def parse_args():

    parser = MyArgParser(prog=None, usage=None, description=None, )

    #### Program args ####
    parser.add_argument("-e", "--n_epochs", type=int, help="Number of epochs to use", default=1000)
    parser.add_argument("-m", "--msg", type=str, help="A message describing the run", required=True)
    parser.add_argument("-r", "--restore", type=str, help="Directory to trained checkpoint to continue training", default=None)
    parser.add_argument("-c", "--config", type=str, help="Path to json file with the config arguments to create/update the "
                                                         "model")
    parser.add_argument("-g", "--gpus", type=int, help="Number of gpus to use", default=1)
    parser.add_argument("-n", "--num_nodes", type=int, help="Number of nodes to use", default=None)
    parser.add_argument("-s", "--random_seed", type=int, help="Random seed", default=121)

    parser.add_argument( "--limit_train_batches", type=float, help="Train only with a fraction of the tranining set", default=None)

    #### Model and data args ####

    group = parser.add_argument_group(title="data")
    GraphPriceDatamodule.addParamsToArgParse(group)
    group = parser.add_argument_group(title="model")
    PricePredictorModule.addParamsToArgParse(group)

    cmd_args = parser.parse_args()
    del cmd_args['positional arguments']
    cmd_args['main'] = cmd_args['optional arguments']
    del cmd_args['optional arguments']
    return cmd_args

if __name__ == "__main__":
    print( " ".join(sys.argv))
    cmd_args = parse_args()

    if cmd_args.get("config"):
        print("Loading configuration file:", cmd_args.get("config"))
        with open(cmd_args.get("config")) as f:
            args = json.load(f)

        for groupName in cmd_args:
            args[groupName].update(cmd_args[groupName])
    else:
        args = cmd_args

    print(args)

    seed_everything(args["main"]["random_seed"])


    if args['main']["restore"]:
        prev_run_dir = os.path.expanduser(args['main']["restore"])
        # hparams = yaml.load( os.path.join(prev_run_dir, "hparams.yaml"))
        checkpointsDir = os.path.join(prev_run_dir, "checkpoints")
        most_recent_checkpoint_fname = max([os.path.join(checkpointsDir, basename) for basename in os.listdir(checkpointsDir)],
                                           key=os.path.getctime)

        pl_model = PricePredictorModule.load_from_checkpoint(most_recent_checkpoint_fname, **args["model"])
        data_args = pl_model.hparams.get("data_hparams")
        data_args.update(args["data"])
        dataModule = GraphPriceDatamodule(**data_args)

        print("Previous model loaded!!")

    else:
        dataModule = GraphPriceDatamodule(**args["data"])

        dataModule.prepare_data()
        nodes_n_features, edges_n_features = dataModule.dims
        nodes_degree = dataModule.get_nodes_degree()
        args["model"].update( dict(
                            nodes_n_features=nodes_n_features, edges_n_features=edges_n_features,
                            deg=nodes_degree, data_hparams= args["data"])
                                   )

        pl_model = PricePredictorModule( **args["model"] )

    print( pl_model.hparams )

    default_root_dir = os.getcwd()
    trainer_args = dict(gpus=1, max_epochs=args['main']["n_epochs"], progress_bar_refresh_rate=20,
                        default_root_dir=default_root_dir, auto_lr_find=True,
              callbacks = [
                       EarlyStopping(monitor='val_loss', patience=60),
                       ModelCheckpoint(monitor='val_loss', verbose=True),
                       PlotFigsTensorboardCallback(frequency=5, save_csv=True)
                           ])
    if args["main"]["limit_train_batches"]:
        trainer_args["limit_train_batches"] = args["main"]["limit_train_batches"]

    ngpus = args['main'].get("gpus", -1)
    if ngpus > 1:
        trainer_args["gpus"] = ngpus
        trainer_args["accelerator"] = "ddp"
        trainer_args["plugins"] = DDPPlugin(find_unused_parameters=False),
        os.environ["NCCL_NSOCKS_PERTHREAD"]="4"
        os.environ["NCCL_SOCKET_NTHREADS"] = "2"
    elif ngpus ==0:
        trainer_args["gpus"] = 0
        trainer_args["accelerator"] = "ddp_cpu"

    if args['main'].get("num_nodes", None):
        trainer_args["num_nodes"] = int(args['main'].get("num_nodes"))
        trainer_args["accelerator"] = "ddp"
        raise NotImplementedError()
        # import torch.distributed as dist
        # dist.init_process_group("gloo", rank=int(os.environ.get("NODE_RANK")), world_size=trainer_args["num_nodes"] )

    try:
        print(trainer_args)
        trainer = pl.Trainer( **trainer_args )
        trainer.fit(pl_model, dataModule)
        res = trainer.test(pl_model, dataModule.test_dataloader(), verbose=True)[0]
        r, __ = InferencePlotter(pl_module=pl_model).pred_and_plot( mode="test")
        res["test_pearson-r"] = r
        print(res)
    finally:
        if args['main'].get("num_nodes", None):
            # dist.destroy_process_group()
            pass