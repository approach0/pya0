import os
import inspect
import datetime
import contextlib
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class BaseTrainer:
    """
    Trainer for typical distributed training process.
    """
    model: 'typing.Any' = None
    dataset_cls: 'typing.Any' = None
    test_loader: 'typing.Any' = None
    test_data_cls: 'typing.Any' = None
    test_file: str = './test.dat'
    test_cycle: int = 50
    test_only: bool = False
    epochs: int = 20
    save_fold: int = 5
    batch_size: int = 2
    cluster: 'typing.Any' = None
    xla_cores: int = 0
    start_point: tuple = (0,0,-1)
    shards_list: str = './shards.txt'
    active_fp16: bool = False
    caller: str = 'nocaller'
    dev_map: tuple = None
    device_ordinal: int = 0

    def infer_start_point(self, save_name):
        if '/' in save_name:
            fields = save_name.strip('/').split('/')[-1].split('-')
            if len(fields) == 3 and all(map(lambda x: x.isdigit(), fields)):
                return tuple(int(v) for v in fields)
        return 0, 0, -1

    def map_device(self, ordinal):
        if self.dev_map is None:
            return ordinal, float('inf')
        else:
            if isinstance(self.dev_map, int):
                map_arr = [self.dev_map]
            else:
                assert isinstance(self.dev_map, tuple)
                map_arr = list(self.dev_map)
            map_arr = list(map(lambda x: int(x), map_arr))
            return map_arr[ordinal], len(map_arr)

    def num_local_dev(self):
        num = 1 # CPU
        if torch.cuda.is_available():
            num = torch.cuda.device_count() # GPU
        elif self.xla_cores > 0:
            num = self.xla_cores # TPU

        _, device_max_num = self.map_device(0)
        return min(num, device_max_num)

    def local_device_info(self):
        n_devices = self.num_local_dev()
        if torch.cuda.is_available():
            ordinal = self.device_ordinal
            device = torch.device('cuda:' + str(ordinal))
            gpu_name = torch.cuda.get_device_name(device)
            gpu_props = torch.cuda.get_device_properties(device)
            gpu_total = gpu_props.total_memory // (1024 ** 2)
            fp16 = ' (FP16)' if self.active_fp16 else ''
            return f"{n_devices} x {gpu_name}{fp16}: {gpu_total}MiB"
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
                shard_files = [
                    os.path.join(dirname, line.rstrip()) for line in fh
                ]
                exists = [os.path.isfile(f) for f in shard_files]
                assert(all(exists))
        except Exception as e:
            print('Error when reading:', self.shards_list)
            quit(1)
        return shard_files

    def prehook(self, *args):
        pass

    def unwrap_model(self):
        if self.cluster:
            return self.model.module # unwrap DDP layer
        else:
            return self.model

    def save_model(self, model, save_funct, save_name, job_id):
        raise NotImplementedError

    def _save_model(self, point, glob_rank, job_id):
        if glob_rank != 0 and self.xla_cores == 0:
            return # must be master node (unless if it is TPU)
        model = self.unwrap_model()
        save_name = ('%d-%d-%d' % point)
        print(f'Saving model "{save_name}" ...')
        if self.xla_cores:
            import torch_xla.core.xla_model as xm
            save_function = xm.save
        else:
            save_function = torch.save
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

    def start_training(self, loop):
        self.model.train()
        self.caller = inspect.stack()[1].function
        print('[caller]', self.caller)
        if self.xla_cores:
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(_train_thread, nprocs=xla_cores, args=(self, loop))
        else:
            import torch.multiprocessing as mp
            n_cores = self.num_local_dev()
            mp.spawn(_train_thread, nprocs=n_cores, args=(self, loop))

    def _prepare_testing(self, mini_batch):
        if self.test_file and self.test_data_cls:
            if not os.path.isfile(self.test_file):
                print('Error: Cannot find test file', self.test_file)
                quit(1)
            test_data = self.test_data_cls(self.test_file)
            print(f'Loading test data: {self.test_file} (bsize={mini_batch})')
            self.test_loader = DataLoader(test_data,
                batch_size=mini_batch,
                collate_fn=lambda batch: batch,
                shuffle=False
            )
            self.test_cnt = 0

    def do_testing(self, eval_func, *args):
        done = False
        if self.test_loader:
            if self.test_cycle != 0 and self.test_cnt % self.test_cycle == 0:
                self.model.eval()
                with torch.no_grad():
                    for test_batch, test_inputs in enumerate(self.test_loader):
                        try:
                            eval_func(test_batch, test_inputs, *args)
                            done = True
                        except StopIteration:
                            break
                self.model.train()
            self.test_cnt += 1
        return done


def _train_thread(local_rank, trainer, loop):
    # hook print function to show node/rank
    import builtins as __builtin__
    def print(*args):
        __builtin__.print(f'[node#{node_id} rank#{glob_rank}]', *args)

    def get_env_var(name, default):
        val = os.environ.get(name)
        return default if val is None else val

    # get cluster information
    n_nodes = int(get_env_var("SLURM_JOB_NUM_NODES", 1))
    node_id = int(get_env_var("SLURM_NODEID", 0))
    job_id = get_env_var("SLURM_JOB_ID", 0)
    n_devices = trainer.num_local_dev()
    glob_batches = n_nodes * n_devices
    glob_rank = node_id * n_devices + local_rank
    is_slave = (glob_rank > 0)
    m_batch = trainer.batch_size // glob_batches

    # move model to device
    if trainer.xla_cores:
        import torch_xla.core.xla_model as xm
        trainer.device_ordinal = 0
        device = xm.xla_device()
    else:
        trainer.device_ordinal, _ = trainer.map_device(local_rank)
        device = torch.device(f'cuda:{trainer.device_ordinal}'
            if torch.cuda.is_available() else 'cpu')

    # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
    # To use DistributedDataParallel on a host with N GPUs, you should
    # spawn up N processes, ensuring that each process exclusively works
    # on a single GPU from 0 to N-1. This can be done by either setting
    # CUDA_VISIBLE_DEVICES for every process or by calling:
    torch.cuda.set_device(trainer.device_ordinal)

    print('Current CUDA device:',  torch.cuda.current_device())
    CUDA_VISIBLE_DEVICES = (
        os.environ['CUDA_VISIBLE_DEVICES' ]
        if 'CUDA_VISIBLE_DEVICES' in os.environ else []
    )
    print('CUDA_VISIBLE_DEVICES:', CUDA_VISIBLE_DEVICES)

    print('Training on device:', device)
    print(trainer.local_device_info())
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
        dist.barrier(device_ids=[int(trainer.device_ordinal)])
        trainer.model = DDP(trainer.model, broadcast_buffers=False,
            device_ids=[torch.cuda.current_device()]
        )
        dist.barrier(device_ids=[int(trainer.device_ordinal)])

    elif trainer.xla_cores:
        import torch_xla.core.xla_model as xm
        print('Enter XLA barrier.')
        xm.rendezvous('init')

    else:
        assert glob_rank == 0, "Multi-GPUs need to specify --cluster option."

    # prehook: setup optimizer etc., after DDP initialization.
    trainer.prehook(device, job_id, glob_rank)

    # Turn on FP16?
    if torch.cuda.is_available() and trainer.active_fp16:
        trainer.scaler = torch.cuda.amp.GradScaler()
        scaler_ctx = torch.cuda.amp.autocast()
    else:
        trainer.scaler = None
        scaler_ctx = contextlib.nullcontext()

    iteration = 0
    shard_files = trainer._get_shard_files()
    n_shards = len(shard_files)
    print('Shards:', shard_files)
    print('Start training at:', trainer.start_point)

    trainer._prepare_testing(m_batch)

    save_cycle = 0
    for epoch in range(trainer.epochs):
        if (epoch,) < trainer.start_point[:1]: continue
        for shard, shard_file in enumerate(shard_files):
            if (epoch, shard) < trainer.start_point[:2]: continue
            print(f'Loading shard {shard_file} ...')
            dataset = trainer.dataset_cls(shard_file)
            # calculating save fold ...
            n_batches = len(dataset) // trainer.batch_size + 1
            if trainer.save_fold == 0 or trainer.test_only:
                save_cycle = 0
            else:
                save_cycle = n_batches // trainer.save_fold
            # prepare dataset loader
            loader = DataLoader(dataset,
                batch_size=trainer.batch_size,
                collate_fn=lambda batch: batch,
                shuffle=True if not trainer.test_only else False
            )
            if trainer.xla_cores:
                import torch_xla.distributed.parallel_loader as pl
                loader = pl.MpDeviceLoader(loader, device)
            # invoke training loop ...
            with tqdm(loader, unit="batch", disable=is_slave) as progress:
                for batch, inputs in enumerate(progress):
                    if (epoch, shard, batch) <= trainer.start_point[:3]:
                        continue
                    bi = slice(glob_rank * m_batch, (glob_rank + 1) * m_batch)
                    inputs = inputs[bi]
                    if len(inputs) == 0:
                        continue # last (incomplete) batch?
                    # invoke train loop
                    args = locals()
                    arg_names = inspect.getargspec(loop)[0]
                    arg_vals = tuple(args[k] if k in args else None
                        for k in arg_names if k != 'self')
                    with scaler_ctx:
                        if trainer.test_only:
                            trainer.model.eval()
                            with torch.no_grad():
                                loop(*arg_vals)
                        else:
                            loop(*arg_vals)
                        iteration += 1
                    # save on cycle
                    if save_cycle > 0 and batch % save_cycle == 0:
                        trainer._save_model(
                            (epoch, shard, batch), glob_rank, job_id
                        )
                        if trainer.xla_cores:
                            import torch_xla.core.xla_model as xm
                            xm.rendezvous('save')
    # final save ...
    if save_cycle > 0:
        trainer._save_model((epoch + 1, 0, 0), glob_rank, job_id)

    # Leaving barrier
    if trainer.cluster:
        dist.destroy_process_group()
