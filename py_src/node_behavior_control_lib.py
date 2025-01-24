from py_src.node import Node
from py_src.simulation_runtime_parameters import RuntimeParameters

def global_broadcast(runtime_parameters: RuntimeParameters, src_node_name: int, mpi_world=None):
    if runtime_parameters.mpi_enabled:
        from mpi4py import MPI
        from mpi4py.util import pkl5
        MPI_comm = MPI.COMM_WORLD
        MPI_rank = MPI_comm.Get_rank()
        MPI_size = MPI_comm.Get_size()
        assert mpi_world is not None, f"mpi_world is None but the runtime_parameters says this is mpi environment"

        MPI_comm.barrier()
        has_src_node = src_node_name in runtime_parameters.node_container.keys()
        all_has_src_node = MPI_comm.gather(has_src_node, root=0)
        # find the node with src model
        node_has_src_node = None
        if MPI_rank == 0:
            node_has_src_node = all_has_src_node.index(True)
        node_has_src_node = MPI_comm.bcast(node_has_src_node, root=0)
        # get the src model
        src_model_stat = None
        if MPI_rank == node_has_src_node:
            src_model_stat = runtime_parameters.node_container[src_node_name].get_model_stat()
        large_comm = pkl5.Intracomm(MPI_comm)
        src_model_stat = large_comm.bcast(src_model_stat, root=node_has_src_node)
        MPI_comm.barrier()
        # set model
        for node_name, node_target in runtime_parameters.node_container.items():
            node_target: Node
            if src_node_name == node_name:
                continue
            node_target.set_model_stat(src_model_stat)

    else:
        assert src_node_name in runtime_parameters.node_container.keys(), f"node {src_node_name} does not exist in the network: {runtime_parameters.node_container.keys()}"
        src_model_stat = runtime_parameters.node_container[src_node_name].get_model_stat()
        for node_name, node_target in runtime_parameters.node_container.items():
            node_target: Node
            if src_node_name == node_name:
                continue
            node_target.set_model_stat(src_model_stat)
