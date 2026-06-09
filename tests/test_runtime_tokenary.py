from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.tokenary import TokenaryRuntime


def test_tokenary_node_command_rendezvous_at_head_with_hosts():
    recipe_data = {
        "name": "test-recipe",
        "model": "/mnt/quant/Minimax-M3-v0-NVFP4",
        "runtime": "tokenary",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = TokenaryRuntime()
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]

    for rank in range(4):
        cmd = runtime.generate_node_command(
            recipe,
            {},
            head_ip="10.0.0.1",
            num_nodes=4,
            node_rank=rank,
            hosts=hosts,
        )
        assert "--master-addr 10.0.0.1" in cmd, "rank %d -> %s" % (rank, cmd)
        assert "--world-size 4" in cmd
        assert "--rank %d" % rank in cmd
