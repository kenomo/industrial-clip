import os
import time
import numpy as np
import datetime
import torch
from tqdm import tqdm
import wandb
from torch.nn import functional as F
from npy_append_array import NpyAppendArray

from dassl.utils import (MetricMeter, AverageMeter)
from dassl.metrics import compute_accuracy
from dassl.engine.trainer import SimpleTrainer
from dassl.utils import (load_checkpoint)

from .evaluator import ExtendedClassification


class TrainerPP(SimpleTrainer):
    """A the trainer class for industrial-clip."""

    def __init__(self, cfg):
        self.init_wandb(cfg)
        super().__init__(cfg)
        
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(tqdm(self.train_loader_x)):
            data_time.update(time.time() - end)

            output = self.forward_backward(batch)

            # check if model also returns image and text features
            if isinstance(output, tuple):
                loss_summary, _, _ = output
            else:
                loss_summary = output
            
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar({
                    "train/" + name: meter.avg,
                    "batch": self.batch_idx
                }, n_iter)
            self.write_scalar({
                "train/lr": self.get_current_lr(),
                "batch": self.batch_idx
            }, n_iter)

            end = time.time()

    def forward_backward(self, batch):

        image, label, idx = self.parse_batch_train(batch)
        # image [Batch Size] -> is the image tensor of batch 
        # label [Batch Size, 3, (Image Size)] -> is the class id tensor of batch
        # idx [Batch Size] -> is an index tensor of batch
        
        output = self.model(image, label)

        # check if model returns also image and text features
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        loss = F.cross_entropy(logits, idx)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, idx)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

         # check if model returns also image and text features
        if isinstance(output, tuple):
            return loss_summary, output[1], output[2]

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        idx = np.arange(len(input))

        input = input.to(self.device)
        label = label.to(self.device)
        idx = torch.from_numpy(idx).to(self.device)

        return input, label, idx

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        idx = np.arange(len(input))

        input = input.to(self.device)
        label = label.to(self.device)
        idx = torch.from_numpy(idx).to(self.device)

        return input, label, idx
        
    def before_train(self):
        directory = self.cfg.OUTPUT_DIR

        if not self.cfg.DO_NOT_RESUME:
            if self.cfg.RESUME:
                directory = self.cfg.RESUME
            self.start_epoch = self.resume_model_if_exist(directory)

    def init_wandb(self, cfg):
        if cfg.WANDB.LOG:
            
            if os.environ['WANDB_API_KEY'] is None:
                raise ValueError("Env var WANDB_API_KEY is not specified")
            
            if cfg.WANDB.PROJECT is None:
                raise ValueError("WANDB.PROJECT is not specified")

            if wandb.run is None:
                # log into wandb
                wandb.login(key=os.environ['WANDB_API_KEY'])

                # start a new wandb run to track training
                if cfg.WANDB.RUN_NAME is None:
                    wandb.init(
                        project=cfg.WANDB.PROJECT,
                        config=cfg,
                    )
                else:
                    wandb.init(
                        project=cfg.WANDB.PROJECT,
                        config=cfg,
                        name=cfg.WANDB.RUN_NAME,
                    )

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # by default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = os.path.join(directory, name, model_file)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            if val_result is None:
                print( f"Load {model_path} to {name} (epoch={epoch})")
            else:
                print( f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})")

            self._models[name].load_state_dict(state_dict)

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # finish run
        if self.cfg.WANDB.LOG:
            wandb.finish()
    
    def after_epoch(self):
        super().after_epoch()
        
        if self.val_loader is not None:
            self.test(split="val")        

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            self.test(split="test")

    def write_scalar(self, dict, global_step=None):
        if self.cfg.WANDB.LOG:
            merged = {**dict, **{"epoch": self.epoch}}
            wandb.log(merged, step=global_step)

    def save_embeddings(self, output, label):
        image_features, text_features = output[1], output[2]
        image_features = image_features.cpu().numpy()
        text_features = text_features.cpu().numpy()
        label = label.cpu().numpy()

        # concatenate image, text features and labels
        data = np.concatenate((image_features, text_features), axis=1)
        data = np.concatenate((data, label[:, None]), axis=1)

        with NpyAppendArray(os.path.join(self.output_dir, "embeddings.npy")) as f:
            f.append(data)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for self.batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, idx = self.parse_batch_test(batch)

            output = self.model(input, label)

            if self.cfg.EVAL.SAVE_EMBEDDINGS:
                self.save_embeddings(output, label)

            # check if model also returns image and text features, discard them during testing
            if isinstance(output, tuple):
                output = output[0]

            # pred = output.max(1)[1]
            # lpred = label[pred].cpu().tolist()
            # spred = [self.lab2cname[i] for i in lpred]
            # matches the ouptut only against the indices in the batch
            self.evaluator.process(output, (idx, label))
        
        results = self.evaluator.evaluate()

        if self.cfg.TEST.EVALUATOR == "ExtendedClassification":

            accuracy = {f"{split}/top-{i + 1}/acc": k for i, k in enumerate(results["topk_accuracy"])}
            total = {f"{split}/top-{i + 1}/total": k for i, k in enumerate(results["topk_total"])}
            correct = {f"{split}/top-{i + 1}/correct": k for i, k in enumerate(results["topk_correct"])}
            self.write_scalar({
                **accuracy, **total, **correct,
                f"{split}/macro_f1": results["macro_f1"]
            })

            if (self.cfg.TEST.PER_CLASS_RESULT and split == "test") or \
               (self.cfg.VALIDATION.PER_CLASS_RESULT and split == "val"):
                self.write_scalar({
                    f"{split}/mean_perclass_accuracy": results["mean_perclass_accuracy"]
                })

            if self.cfg.TEST.PER_CLASS_RESULT and split == "test":
                print("\n".join(results["perclass_accuracy"]))
            
            if ((self.cfg.TEST.COMPUTE_CMAT and split == "test") or \
                (self.cfg.VALIDATION.COMPUTE_CMAT and split == "val")) and \
                (self.cfg.WANDB.LOG):
                cm = wandb.plot.confusion_matrix(
                    y_true=results["y_true"],
                    preds=results["y_pred"],
                    class_names=results["classnames"]
                )
                wandb.log({f"{split}/conf_mat": cm})

        return list(results.values())[0]
    
    @torch.no_grad()
    def forward(self, input, label):
        self.set_model_mode("eval")

        output = self.model(input, label)
        # check if model also returns image and text features, discard them during testing
        if isinstance(output, tuple):
            output = output[0]
        
        return output