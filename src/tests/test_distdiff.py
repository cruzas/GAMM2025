import diffdist.functional as distops # pip install diffdist
import torch.distributed as dist
import torch
import diffdist.extra_collectives as extra_comm
import os, subprocess
import torch.multiprocessing as mp

def prepare_distributed_environment(rank, master_addr, master_port, world_size):
    device_id = 0
    if rank is None and master_addr is None and master_port is None and world_size is None: # we are on a cluster
        ## Execute code on a cluster
        os.environ["MASTER_PORT"] = "29501"
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NNODES"]
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = os.environ["SLURM_NODEID"]
        node_list = os.environ["SLURM_NODELIST"]
        master_node = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        os.environ["MASTER_ADDR"] = master_node
        dist.init_process_group(backend="nccl")
    else: # we are on a PC
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port # A free port on the master node
        # os.environ['WORLD_SIZE'] = str(world_size) # The total number of GPUs in the distributed job
        # os.environ['RANK'] = '0' # The unique identifier for this process (0-indexed)
        # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo" # "nccl" or "gloo"
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    device_id = dist.get_rank()
    print(f"Device id: {device_id}")

def test_reduce_scatter():
    if dist.get_rank() == 0:
        print("REDUCE_SCATTER TEST\n")
    x = torch.arange(dist.get_world_size()).float().split(1)
    buff = torch.tensor(0.)
    extra_comm.reduce_scatter(buff, x)
    print(dist.get_rank(), x)
    print(dist.get_rank(), buff)
    dist.barrier()
    if dist.get_rank() == 0:
        print('-' * 50)


def test_all_gather():
    if dist.get_rank() == 0:
        print("ALL GATHER TEST\n")
    dist.barrier()
    x = torch.tensor(3., requires_grad=True)
    y = (dist.get_rank() + 1) * x

    print(dist.get_rank(), "Sending y:", y)
    z = distops.all_gather(list(torch.zeros(dist.get_world_size())),
                           y,
                           next_backprop=None,
                           inplace=True)
    print(dist.get_rank(), "Received tensor:", z)
    l = torch.sum(torch.stack(z))
    l = l * (dist.get_rank() + 1)
    l.backward()

    print(dist.get_rank(), "Gradient with MPI:", x.grad)
    dist.barrier()
    if dist.get_rank() == 0:
        print()
        x = [
            torch.tensor(3., requires_grad=True)
            for i in range(dist.get_world_size())
        ]
        res = []
        for i in range(1, dist.get_world_size() + 1):
            res.append(i * x[i - 1])

        res2 = []
        for i in range(dist.get_world_size()):
            temp = []
            for j in range(dist.get_world_size()):
                temp.append(torch.clone(res[j]))
            res2.append(temp)
        l_s = [torch.sum(torch.stack(i)) for i in res2]
        final = [(i + 1) * k for i, k in enumerate(l_s)]
        for i in range(dist.get_world_size() - 1):
            final[i].backward(retain_graph=True)
        final[-1].backward()
        for i, x_i in enumerate(x):
            print(i, "Gradient in single process:", x_i.grad)
        print('-' * 50)


def test_scatter():
    if dist.get_rank() == 0:
        print("SCATTER TEST\n")
        x = [
            torch.tensor(3., requires_grad=True)
            for i in range(dist.get_world_size())
        ]
        y = [2 * x_i for x_i in x]

        print("Sending y:", y)
        buffer = torch.tensor(0.)
        z = distops.scatter(buffer, y, src=0, inplace=False)
    else:
        buffer = torch.tensor(0., requires_grad=True)
        z = distops.scatter(buffer, src=0, inplace=False)

    print(dist.get_rank(), "Received tensor:", z)
    # Computation
    k = (dist.get_rank() + 1) * z
    k.backward()

    if dist.get_rank() == 0:
        print("Gradient with MPI:", [x_i.grad for x_i in x])

    if dist.get_rank() == 0:
        print()
        x = [
            torch.tensor(3., requires_grad=True)
            for i in range(dist.get_world_size())
        ]
        y = [2 * x_i for x_i in x]
        res = []
        for i in range(dist.get_world_size()):
            res.append((i + 1) * y[i])

        for i, k in enumerate(res):
            k.backward()
        print("Gradient in single process:", [x_i.grad for x_i in x])
    dist.barrier()
    if dist.get_rank() == 0:
        print('-' * 50)


def test_gather():
    if dist.get_rank() == 0:
        print("GATHER TEST\n")
    dist.barrier()
    x = torch.tensor(3., requires_grad=True)
    y = (dist.get_rank() + 1) * x

    print(dist.get_rank(), "Sending y:", y)
    if dist.get_rank() == 0:
        z = distops.gather(y,
                           torch.zeros(dist.get_world_size()).split(1),
                           dst=0,
                           next_backprop=None,
                           inplace=True)
        print(dist.get_rank(), "Received tensor:", z)
        l = torch.sum(torch.stack(z))
        l.backward()
    else:
        dummy = distops.gather(y, dst=0, next_backprop=None, inplace=True)
        dummy.backward(torch.tensor([]))
    print(dist.get_rank(), "Gradient with MPI:", x.grad)
    dist.barrier()
    if dist.get_rank() == 0:
        print()
        x = [
            torch.tensor(3., requires_grad=True)
            for i in range(dist.get_world_size())
        ]
        res = []
        for i in range(1, dist.get_world_size() + 1):
            res.append(i * x[i - 1])

        z = torch.stack(res)
        l = torch.sum(z)
        l.backward()
        for i, x_i in enumerate(x):
            print(i, "Gradient in single process:", x_i.grad)
        print('-' * 50)


def test_broadcast():
    if dist.get_rank() == 0:
        print("BROADCAST TEST\n")
        x = torch.tensor(3., requires_grad=True)
        y = 2 * x

        print(dist.get_rank(), "Sending y:", y)
        z = distops.broadcast(y, src=0, inplace=False)
        print(dist.get_rank(), "Received tensor:", z)

        # Computation
        k = 3 * z
        k.backward()
        print("Gradient with MPI:", x.grad)

        print()
        x = torch.tensor(3., requires_grad=True)
        y = 2 * x
        res = [3 * y]
        for i in range(1, dist.get_world_size()):
            res.append(9 * y)

        for i, k in enumerate(res):
            if i == (len(res) - 1):
                k.backward()
            else:
                k.backward(retain_graph=True)
        print("Gradient in single process:", x.grad)
    else:
        x = torch.tensor(5., requires_grad=True)
        y = 7 * x

        buffer = torch.tensor(0.)
        z = distops.broadcast(buffer, src=0, next_backprop=y)
        print(dist.get_rank(), "Received tensor:", z)
        k = 9 * z
        k.backward()
        print(dist.get_rank(), "Grad of disconnected part:", x.grad)
    dist.barrier()
    if dist.get_rank() == 0:
        print('-' * 50)


def test_consume_variable():
    x = torch.tensor(5., requires_grad=True)
    y = 2 * x

    z = 3 * y
    j = 4 * y

    z = distops.consume_variable(j, [z], set_ones_grad=True)[0]
    print(z)
    z.backward()
    print(x.grad)
    print()
    x = torch.tensor(5., requires_grad=True)
    y = 2 * x

    z = 3 * y
    j = 4 * y

    z.backward(retain_graph=True)
    j.backward()
    print(x.grad)


def test_send_recv(rank, master_addr, master_port, world_size):
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    
    if dist.get_rank() == 0:
        print("SEND/RECV TEST\n")
        x = torch.tensor(3., requires_grad=True)
        y = 2 * x

        print("Before sending y:", y)
        connector = distops.send(y, dst=1)
        # Computation happens in process 1
        buffer = torch.tensor(0.)
        z, _ = distops.recv(buffer, src=1, next_backprop=connector)
        print("After receiving:", z)

        k = 3 * z
        k.backward()
        print("Gradient with MPI:", x.grad)

        print()
        x = torch.tensor(3., requires_grad=True)
        y = 2 * x
        l = y * 10
        k = 3 * l
        k.backward()
        print("Gradient in single process:", x.grad)
        print('-' * 50)
    elif dist.get_rank() == 1:
        buffer = torch.tensor(0., requires_grad=True)
        y, _ = distops.recv(buffer, src=0)

        l = y * 10

        connector = distops.send(l, dst=0)
        connector.backward(torch.tensor([]))


if __name__ == '__main__':
    torch.manual_seed(1)

    world_size = torch.cuda.device_count()  
    master_addr = 'localhost'
    master_port = '12345'
    mp.spawn(test_send_recv, args=(master_addr, master_port, world_size), nprocs=world_size, join=True)


    # test_broadcast()

    # test_gather()

    # test_scatter()

    # test_all_gather()