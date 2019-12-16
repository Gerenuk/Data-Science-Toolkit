import itertools
import re
import pandas as pd
from IPython.display import display
from pathlib import Path


class DataFlowTracker:
    def __init__(self):
        self.data_flow_info = []

    def __call__(self, func):
        def wrapped(*args):
            input_info = self.encode_data(args)

            print(f"Running {func.__name__}")
            result = func(*args)

            output_info = self.encode_data(result)

            self.add_info(
                {"input": input_info, "output": output_info, "name": func.__name__}
            )

            return result

        return wrapped

    def encode_data(self, data, default_name=None):
        if isinstance(data, (list, tuple)):
            return list(
                itertools.chain.from_iterable(self.encode_data(el) for el in data)
            )

        if isinstance(data, (pd.DataFrame, pd.Series)):
            if isinstance(data, pd.Series):
                data = data.to_frame()

            return [
                {
                    "type": "dataframe",
                    "columns": list(data.columns),
                    "name": data.name if hasattr(data, "name") else "_noname_",
                    "num_rows": len(data),
                    "teile": data["teil"].unique() if "teil" in data else None,
                }
            ]  # !!! default_name?

        if isinstance(data, Path):
            return [{"type": "text", "name": data}]

        return []

    def add_info(self, info):
        self.data_flow_info.append(info)

    def create_dot_graph(self, node_label_func, transform_label_func):
        lines = ["digraph dataflow {"]

        nodes = {}
        links = []

        def add_node(name, info):
            if name not in nodes:
                nodes[name] = info

        def calc_dot_name(data):
            if data["type"] == "dataframe":
                return data["name"]
            if data["type"] == "text":
                name = data["name"].name
                name = re.sub("[^A-Za-z0-9]", "_", name)
                return name

        for item in self.data_flow_info:
            for data in item["input"] + item["output"]:
                if data["type"] == "dataframe":
                    add_node(
                        data["name"],
                        {
                            "label": node_label_func(data),
                            "shape": "box",
                            "style": "filled",
                            "fillcolor": '"#fedd8a"',
                        },
                    )
                elif data["type"] == "text":
                    add_node(calc_dot_name(data), {"label": node_label_func(data), "shape": "box", "style": "filled", "fillcolor": '"#d3d3d3"'})
                else:
                    raise ValueError(item)

            link_name = item["name"]
            add_node(
                str(link_name),
                {"label": transform_label_func(item), "style": "filled", "fillcolor": '"#db9dc9"'},
            )

            for data in item["input"]:
                links.append((calc_dot_name(data), link_name))

            for data in item["output"]:
                links.append((link_name, calc_dot_name(data)))

        for name, node in nodes.items():
            lines.append(
                f"{name} ["
                + " ".join(
                    f"{key}={val}" for key, val in node.items() if key != "name"
                )
                + "];"
            )

        for link_from, link_to in links:
            lines.append(f"{link_from} -> {link_to};")

        lines.append("}")

        return "\n".join(lines)

    def show(self, node_label_func, transform_label_func):
        dot = self.create_dot_graph(node_label_func, transform_label_func)
        # print(dot)
        display({"text/vnd.graphviz": dot}, raw=True)

    def write_graph(self, filename, node_label_func, transform_label_func):
        with open(filename, "w") as f:
            f.write(self.create_dot_graph(node_label_func, transform_label_func))
