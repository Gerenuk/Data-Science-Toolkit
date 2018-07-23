def print_tree(root):
    print(_tree_show_str(root._tree_show_node()))


def _reformat_tree_show(text, first_prepend="", next_prepends=""):
    lines = text.split("\n")
    lines = [first_prepend + lines[0]] + [next_prepends + line for line in lines[1:]]
    return "\n".join(lines)


def _tree_show_str(node, chain_depth=0):
    """
    Double pipe means it is a linear connection
    Single pipe means it is one of multiple inputs
    """
    show_parts = []

    name = node.name
    num_children = len(node.children)

    if num_children == 0:
        if chain_depth == 0:
            show_parts.append("═ " + name)
        else:
            show_parts.append("╚ " + name)
    elif num_children == 1:
        if chain_depth == 0:
            show_parts.append("╦ " + name)
            show_parts.append(_reformat_tree_show(_tree_show_str(node.children[0], chain_depth + 1)))
        else:
            show_parts.append("╠ " + name)
            show_parts.append(_reformat_tree_show(_tree_show_str(node.children[0], chain_depth + 1)))
    else:
        if chain_depth == 0:
            show_parts.append("═ " + name)
        else:
            show_parts.append("╚ " + name)

        for i, child in enumerate(node.children):
            if i < len(node.children) - 1:
                show_parts.append(_reformat_tree_show(_tree_show_str(child), "  ├─", "  │ "))
            else:
                show_parts.append(_reformat_tree_show(_tree_show_str(child), "  └─", "    "))

    return "\n".join(show_parts)
