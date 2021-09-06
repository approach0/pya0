import os
import GPUtil
import inspect
import datetime
import contextlib
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


def get_env_var(name, default):
    val = os.environ.get(name)
    return default if val is None else int(val)


@dataclass
class BaseTrainer:
    """
    Trainer for typical distributed training process.
    """
    model: 'typing.Any' = None
    dataset_cls: 'typing.Any' = None
    epochs: int = 20
    save_fold: int = 5
    batch_size: int = 2
    cluster: 'typing.Any' = None
    xla_cores: int = 0
    start_point: tuple = (0,0,-1)
    shards_list: str = './shards.txt'
    active_fp16: bool = False
    caller: str = 'nocaller'

    def infer_start_point(self, save_name):
        if '/' in save_name:
            fields = save_name.strip('/').split('/')[-1].split('-')
            if len(fields) == 3 and all(map(lambda x: x.isdigit(), fields)):
                return tuple(int(v) for v in fields)
        return 0, 0, -1

    def num_local_dev(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count() # GPU
        elif self.xla_cores > 0:
            return self.xla_cores # TPU
        else:
            return 1 # CPU

    def local_device_info(self):
        n_devices = self.num_local_dev()
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_name = gpu.name
            gpu_total = int(gpu.memoryTotal // 1000)
            gpu_load = int(gpu.load * 100)
            gpu_temp = int(gpu.temperature)
            fp16 = ' (FP16)' if self.active_fp16 else ''
            return f"{n_devices} x {gpu_name}{fp16}: {gpu_load}%"
        elif xla_cores:
            import torch_xla
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            dev_info = torch_xla.core.xla_model.get_memory_info(dev)
            total_mem = dev_info['kb_total'] / 1000
            return f'{n_devices} x TPU core {total_mem}MiB'
        else:
            return 'CPU'

    def _get_shard_files(self):
        try:
            assert os.path.isfile(self.shards_list)
            dirname = os.path.dirname(self.shards_list)
            with open(self.shards_list, 'r') as fh:
                shard_files = [dirname + '/' + line.rstrip() for line in fh]
                exists = [os.path.isfile(f) for f in shard_files]
                assert(all(exists))
        except Exception as e:
            print(f'Error: {self.shards_list} exists?')
            quit(1)
        return shard_files

    def prehook(self, device):
        pass

    def save_model(self, model, save_funct, save_name, job_id):
        raise NotImplementedError

    def _save_model(self, point, glob_rank):
        if glob_rank != 0 and self.xla_cores == 0:
            return # must be master node (unless if it is TPU)
        if self.cluster:
            model = self.model.module # unwrap DDP layer
        else:
            model = self.model
        save_name = ('%d-%d-%d' % point)
        print(f'Saving model "{save_name}" ...')
        if self.xla_cores:
            import torch_xla.core.xla_model as xm
            save_function = xm.save
        else:
            save_function = torch.save
        job_id = get_env_var("SLURM_JOB_ID", 0)
        self.save_model(model, save_function, save_name, job_id)

    def backward(self, loss, **args):
        if self.scaler:
            self.scaler.scale(loss).backward(**args)
        else:
            loss.backward(**args)

    def step(self):
        if self.xla_cores:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(self.optimizer)
        elif self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def start_training(self, train_loop):
        self.caller = inspect.stack()[1].function
        print('[caller]', self.caller)
        if self.xla_cores:
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(_train_thread, nprocs=xla_cores, args=(self, train_loop))
        else:
            import torch.multiprocessing as mp
            n_cores = self.num_local_dev()
            mp.spawn(_train_thread, nprocs=n_cores, args=(self, train_loop))


def _train_thread(local_rank, trainer, train_loop):
    # hook print function to show node/rank
    import builtins as __builtin__
    def print(*args):
        __builtin__.print(f'[node#{node_id} rank#{glob_rank}]', *args)

    # get cluster information
    n_nodes = get_env_var("SLURM_JOB_NUM_NODES", 1)
    node_id = get_env_var("SLURM_NODEID", 0)
    n_devices = trainer.num_local_dev()
    glob_batches = n_nodes * n_devices
    glob_rank = node_id * n_devices + local_rank
    is_slave = (glob_rank > 0)

    # move model to device
    if trainer.xla_cores:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(f'cuda:{local_rank}'
            if torch.cuda.is_available() else 'cpu')
    print('Training on device', device)
    trainer.model.to(device)

    # Cluster/XLA barrier
    if trainer.cluster:
        print(f'Initialized process group ...')
        dist.init_process_group(
            backend="nccl", # can be mpi, gloo, or nccl
            init_method=trainer.cluster,
            world_size=glob_batches,
            rank=glob_rank,
            timeout=datetime.timedelta(0, 5 * 60) # 5min timeout
        )
        print('Enter Torch DDP.')
        dist.barrier()
        trainer.model = DDP(trainer.model)
        dist.barrier()
    elif trainer.xla_cores:
        import torch_xla.core.xla_model as xm
        print('Enter XLA barrier.')
        xm.rendezvous('init')

    # prehook: setup optimizer etc., after DDP initialization.
    trainer.prehook(device)

    # Turn on FP16?
    if torch.cuda.is_available() and trainer.active_fp16:
        trainer.scaler = torch.cuda.amp.GradScaler()
        scaler_ctx = torch.cuda.amp.autocast()
    else:
        trainer.scaler = None
        scaler_ctx = contextlib.nullcontext()

    shard_files = trainer._get_shard_files()
    n_shards = len(shard_files)
    print('Shards:', shard_files)

    print('Start training at:', trainer.start_point)
    for epoch in range(trainer.epochs):
        if (epoch,) < trainer.start_point[:1]: continue
        for shard, shard_file in enumerate(shard_files):
            if (epoch, shard) < trainer.start_point[:2]: continue
            print(f'Loading shard {shard_file} ...')
            dataset = trainer.dataset_cls(shard_file)
            # calculating save fold ...
            n_batches = len(dataset) // trainer.batch_size + 1
            if trainer.save_fold == 0:
                save_cycle = 0
            else:
                save_cycle = n_batches // trainer.save_fold
            # prepare dataset loader
            loader = DataLoader(dataset,
                batch_size=trainer.batch_size,
                collate_fn=lambda batch: batch,
                shuffle=True # each shard should shuffle
            )
            if trainer.xla_cores:
                import torch_xla.distributed.parallel_loader as pl
                loader = pl.MpDeviceLoader(loader, device)
            # invoke training loop ...
            with tqdm(loader, unit="batch", disable=is_slave) as progress:
                for batch, inputs in enumerate(progress):
                    if (epoch, shard, batch) <= trainer.start_point[:3]:
                        continue
                    bb = trainer.batch_size // glob_batches
                    bi = slice(glob_rank * bb, (glob_rank + 1) * bb)
                    inputs = inputs[bi]
                    if len(inputs) == 0:
                        continue # last (incomplete) batch?
                    # invoke train loop
                    args = locals()
                    arg_names = inspect.getargspec(train_loop)[0]
                    arg_vals = tuple(args[k] for k in arg_names if k != 'self')
                    with scaler_ctx:
                        train_loop(*arg_vals)
                    # save on cycle
                    if save_cycle > 0 and batch % save_cycle == 0:
                        trainer._save_model((epoch, shard, batch), glob_rank)
                        if trainer.xla_cores:
                            import torch_xla.core.xla_model as xm
                            xm.rendezvous('save')
    # final save ...
    if save_cycle > 0:
        trainer._save_model((epoch + 1, 0, 0), glob_rank)

    # Leaving barrier
    if trainer.cluster:
        dist.destroy_process_group()
