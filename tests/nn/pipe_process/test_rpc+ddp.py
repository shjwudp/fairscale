import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import rpc

from fairscale.nn.pipe import PipeRPCWrapper
from fairscale.utils.testing import get_worker_map, torch_spawn
import fairscale.nn.model_parallel as mpu


def initialize_model_parallel_utility_interface(
    model_parallel_size,
    pipeline_length,
):
    mpu.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    rpc.shutdown()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10639"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(
        model_parallel_size,
        pipeline_length,
        ddp_backend="gloo")
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

    # init rpc
    os.environ['MASTER_PORT'] = "10640"
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    rpc.init_rpc(
        f"Test{torch.distributed.get_rank()}",
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
    )


class DataParallelWork:

    def __init__(self, module, dp_group) -> None:
        self.module = module
        self.dp_group = dp_group
        self._reference_global_rank = dist.distributed_c10d._get_global_rank(self.dp_group, 0)

        for t in self.module.state_dict().values():
            dist.broadcast(
                t,
                src=self._reference_global_rank,
                group=self.dp_group)

    def allreduce_gradients(self):
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(self.dp_group.size())
                dist.all_reduce(p.grad, group=self.dp_group)


def register_optimizer(ctx, model):
    model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.ddp = DataParallelWork(model, mpu.get_data_parallel_group())


def zero_grad(ctx, model):
    model.optimizer.zero_grad()


def step_optimizer(ctx, model):
    model.ddp.allreduce_gradients()
    model.optimizer.step()


@torch_spawn([6])
def rpc_pipe_and_data_parallel():
    model = [
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    ]
    balance = [2, 3, 3]

    initialize_model_parallel_utility_interface(
        model_parallel_size=1,
        pipeline_length=3,
    )
    dp_group = mpu.get_data_parallel_group()
    dp_rank0 = dist.distributed_c10d._get_global_rank(dp_group, 0)

    # Pipeline parallelism rank-0 initialize pipeline module and
    # operate training, other ranks holding rpc service.
    if mpu.get_pipeline_parallel_group().rank() == 0:
        pipe = PipeRPCWrapper(model, balance, worker_map=get_worker_map())

        pipe.foreach_worker(register_optimizer, include_self=True)

        def step():
            inputs = torch.rand(10).cuda()
            target = torch.rand(10).cuda()

            pipe.foreach_worker(zero_grad, include_self=True)

            output = pipe(inputs)

            nn.MSELoss()(output, target).backward()

            pipe.foreach_worker(step_optimizer, include_self=True)

        pipe.train()
        step()

        pipe.eval()
        inputs = torch.rand(10).cuda()

        # Sync input for all pipeline module, expect the same output from them.
        dist.broadcast(inputs, src=dp_rank0, group=dp_group)
        output = pipe(inputs)
        tensor_list = [torch.zeros_like(output, dtype=output.dtype) for _ in range(dp_group.size())]
        dist.all_gather(tensor_list, output, group=dp_group)
        for i in range(1, dp_group.size()):
            assert torch.equal(tensor_list[i], tensor_list[0]), \
                f"tensor_list[{i}]={tensor_list[i]}, tensor_list[0]={tensor_list[0]}"

        torch.distributed.barrier()
        rpc.shutdown()
    else:
        torch.distributed.barrier()
        rpc.shutdown()
