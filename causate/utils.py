import networkx as nx
supported_algorithms= ["PC"]
supported_logging_target=["mlflow_server",
                          "databricks_uc"]

graph_layout_functions = {
            "spring": nx.spring_layout,
            # "circular": nx.circular_layout,
            # "kamada_kawai": nx.kamada_kawai_layout,
            "random": nx.random_layout,
            # "shell": nx.shell_layout,
        }

