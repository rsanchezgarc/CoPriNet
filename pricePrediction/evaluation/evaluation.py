import os, io
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pytorch_lightning.callbacks import Callback

def plt_to_numpy(fig):
    #it should be called after having done the plt.plot stuff, in replacement of plt.show()
    #TO be used with torch tensorboard
    from skimage.color import rgba2rgb
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=fig.dpi)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    img_arr = rgba2rgb( img_arr )
    img_arr = img_arr.astype(np.float32)
    io_buf.close()
    return img_arr

class InferencePlotter():
    def __init__(self, pl_module: 'pl.LightningModule'):
        self.pl_module = pl_module
        self.data_loaders = {}

    def get_dataloader(self, mode, n_workers=None):

        if mode in self.data_loaders:
            return self.data_loaders[mode]
        else:
            if mode == "train":
                dataloader = self.pl_module.train_dataloader(num_workers=n_workers)
            elif mode == "val":
                dataloader = self.pl_module.val_dataloader(num_workers=n_workers)
            elif mode == "test":
                dataloader = self.pl_module.test_dataloader(num_workers=n_workers)
            else:
                raise ValueError("Error. Option mode=%s not supported" % mode)

            self.data_loaders[mode] = dataloader
            return dataloader

    @classmethod
    def do_plots(cls, ys, y_preds, label="test", tensorboard_wrapper=None, step=0):

        if tensorboard_wrapper:
            figArgs = dict(figsize=(5, 4), dpi=450)
        else:
            figArgs = {}
        fig = plt.figure(**figArgs)
        plt.hist(ys, bins=200)
        name = label + " price hist"
        plt.title(name)
        img_arr = plt_to_numpy(fig)
        if tensorboard_wrapper:
            tensorboard_wrapper.add_image(name, img_arr.transpose([2, 0, 1]), step)
            plt.close()
        else:
            plt.show()

        if y_preds is not None:
            fig = plt.figure(**figArgs)
            plt.hist(y_preds, bins=200)
            name = label + " pred hist"
            plt.title(name)
            img_arr = plt_to_numpy(fig)
            if tensorboard_wrapper:
                tensorboard_wrapper.add_image(name, img_arr.transpose([2, 0, 1]), step)
                plt.close()
            else:
                plt.show()

            r, p = pearsonr(ys, y_preds)
            rho, p2 = spearmanr(ys, y_preds)
            if tensorboard_wrapper:
                tensorboard_wrapper.add_scalar(label + '_pearson-r', r, step)
                tensorboard_wrapper.add_scalar(label + '_spearman-r', rho, step)

                if label == "test":
                    print("pearson-r", r, "p-val", p)
                    print("spearman-r", rho, "p-val", p)
            else:
                print("pearson-r", r, "p-val", p)
                print("spearman-r", rho, "p-val", p)

            fig = plt.figure(**figArgs)
            max_val = max([9, ys.max()])
            min_val = 1
            plt.hexbin(ys, y_preds, bins="log", gridsize=100, extent=[min_val, max_val, min_val, max_val] )
            plt.axline((1,1), slope=1, c="r", lw=1, ls="--", alpha=0.5)
            plt.xlabel("log(price)")
            plt.ylabel("Pred")
            plt.colorbar()
            name = label + " price vs pred"
            plt.title(name)
            img_arr = plt_to_numpy(fig)

            if tensorboard_wrapper:
                tensorboard_wrapper.add_image(name, img_arr.transpose([2, 0, 1]), step)
                plt.close()
            else:
                plt.show()

            # fig = plt.figure(**figArgs)
            # plt.boxplot( abs(y_preds - ys)/ys )
            # name = method_name + "_relative_error_boxplot"
            # plt.title(name)
            # img_arr = plt_to_numpy(fig)
            # if tensorboard_wrapper:
            #     tensorboard_wrapper.add_image(name, img_arr.transpose([2,0,1]), step )
            #     plt.close()
            # else:
            #     plt.show()
            return r

    def get_ytrue_ypred(self, mode="test"):
        with torch.no_grad():
            ytrue_ypred = []
            dataLoader = self.get_dataloader(mode=mode)
            n_elems = len(dataLoader)
            for i, (batch, y) in enumerate(dataLoader):
                ytrue_ypred.append((y, self.pl_module(batch.to(self.pl_module.device))))
                if i % 10 == 0: print("%07d/%07d            " % (i,n_elems), end="\r")
            print()
            ytrue_train, ypred_train = zip(*ytrue_ypred)
            ytrue = torch.cat(ytrue_train).to("cpu").numpy()
            ypred = torch.cat(ypred_train).to("cpu").numpy()
        return ytrue, ypred

    def pred_and_plot(self, mode="test", step=0):
        ytrue, ypred = self.get_ytrue_ypred(mode=mode)
        r = self.do_plots(ytrue, ypred, label=mode,
                          tensorboard_wrapper=self.pl_module.logger.experiment,
                          step=step)
        return r, (ytrue, ypred)


class PlotFigsTensorboardCallback(Callback):

    def __init__(self, output_dir: str = None, frequency=10, save_csv=True):
        super().__init__()
        self.output_dir = output_dir
        self.frequency = frequency
        self.current_epoch = 0
        self.plotter = None  # InferencePlotter(pl_module=)
        self.save_csv = save_csv

    def setup(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', stage=None) -> None:
        self.plotter = InferencePlotter(pl_module=pl_module)

    # def on_init_end(self, trainer):
    #     if self.output_dir is None:
    #         self.output_dir = trainer.log_dir

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if trainer.is_global_zero:
            if self.output_dir is None:
                self.output_dir = trainer.log_dir
            if self.save_csv:
                self.csv_fname = os.path.join(self.output_dir, "logs.csv")
                with open(self.csv_fname, "w") as f:
                    f.write("Epoch,Date,trainL1,valL1,trainR,valR")

    def _getDateHourStr(self):
        return datetime.now().strftime("%d/%m-%H:%M")

    def on_train_epoch_end(
            self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused=None
    ) -> None:

        if trainer.is_global_zero:
            self.current_epoch += 1
            if self.current_epoch % self.frequency != 0:
                return
            print("Evaluating preds on epoch %d [%s]" % (self.current_epoch - 1, self._getDateHourStr()))
            r_train, lossL1_train = self.process_data(mode="train")
            r_val, lossL1_val = self.process_data(mode="val")

            if self.save_csv:
                with open(self.csv_fname, "a") as f:
                    f.write("\n%d,%s,%f,%f,%1.4f,%1.4f" % (self.current_epoch - 1, self._getDateHourStr(),
                                                           lossL1_train, lossL1_val, r_train, r_val,))

    def process_data(self, mode="train"):
        r, (ytrue, ypred) = self.plotter.pred_and_plot(mode=mode, step=self.current_epoch)
        lossL1 = float(np.mean(np.abs(ytrue - ypred)))
        return r, lossL1


if __name__ == "__main__":
    import numpy as np

    ys = 1 + np.random.rand(4000) * 7
    y_preds = ys - np.random.rand()
    InferencePlotter.do_plots(ys, y_preds, label="test", tensorboard_wrapper=None)