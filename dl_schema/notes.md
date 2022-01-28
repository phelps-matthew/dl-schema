## Tensorboard vs MLFlow
* Tensorboard has better parallel plot filtering (log scaling, quartile, and bounds settings), however MLflow better handles non-standard input types, like python functions.
* Both mlflow and tb support showing images, with better refresh rate from tb
* Much easier to build parallel coordinates plot in mlf as it is simply a derived outcome of comparing runs with logged parameters. tb requires passing an ugly add_hparams
* MLflow has advantage of logging artifacts which are more arbitrarily viewable compared to tb.
* mlf params can easily be downloaded as a json
* mlf runs can be relatively easily analyzed as dataframes (or exported) using their non-ui api. Good for post-analysis
* mlf runs allow you to view all artifacts logged, which is nice. 
* mlf has superior ability of creating individual experiments, runs, search support based on params, and comparison of aribtary runs

## Other
* Note: Removed clip grad norm from Karpathy implementation. May be useful later.
* 'Simplified' training via methods of train_epoch and test_epoch. Easier to manage these loops.

## PyTorch loss forward function warning.
* This occurs from loss being a scalar and not a torch tensor. However, to convert to tensor one needs to attach to the correct device. This could potentially be inferred from inputs (x.device), though it could be possible that the device of x may differ (non standard however).
* Even when converting to the same device, there is still an issue with respect to gradients.
* Specifically: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
* Since pytorch converts these scalars to a tensor anyway, we are going to just leave as is and deal wwith the warning.


## Ray MLFlow mixin vs callback
* Callback logs more Ray specific parameters like times, iterations since last restore, time total, etc. However, it stores them as metrics (dumb).
* In addition, callbacks archive all the underlying mlflow logs within a separate directory under the same run. This means you cannot view the metrics that are not reported directly to tune, though they are saved as files.
* While some of these timing parameters could be useful, it looks much more appealing to simply use mlflow as designed and with minimal changes required. Viewing metrics for each run is useful. 
* As a result, will be using mixin instead of callback. 

## Configuration managers
* pyrallis - can be used as dataclass, yaml, or auto generation of CLI args. Also support for encoding/decoding quantities to yaml/json. nice dataclass nesting
* yann - underdeveloped, and also quite large as a package. scope too big
* yacs - good, but does not use python dataclasses
* dcargs - close to pyrallis, no support for registering encoders/decoders. not as good cli auto gen for  nested dataclasses
