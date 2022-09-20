def add_time_indexed_parameters_to_block(parameters, block1, block2, time_points):
    seen = set()
    for comp in parameters:
        cuid = get_time_indexed_cuid(parameters)
        comp = block1.find_component(cuid)
        for t in time_points:
            data = comp[t]
            if id(data) in seen:
                continue
            parent = data.parent_block()
            parent_on_2 = block2.find_component(parent)
            local_comp = data.parent_component()
            name = local_comp.local_name
            local_index = local_comp.index_set()
            # Replace time in index set if applicable
            param_on_2 = Param(index_set, mutable=True)
            parent_on_2.add_component(name, param_on_2)
            for sibling_compdata in local_comp.values():
                seen.add(id(sibling_compdata))


def get_series_from_parameters_in_block(parameters, block, time_points):
    data = {}
    for param1 in parameters:
        cuid = get_time_indexed_cuid(param1)
        param2 = block.find_component(cuid)
        data[cuid] = [param2[t] for t in time_points]
    series = TimeSeriesData(data, time_points)
    return series


def copy_block_tree(block1, block2, set_map=None, active=True, descend_index=None):
    """block1 -> block2

    This assumes that block1 is "regular" wrt sets in set_map, i.e. its
    subblocks at different indices of these sets have the same block trees.

    """
    # How do I want to structure this functionality?
    # generate names of blocks to add along with their index sets?
    if set_map is None:
        set_map = ComponentMap()
    for b in block1.component_objects(Block, active=active, descend_into=False):
        local_parent = b.parent_block
        sets = b.index_set().subsets()


def add_blocks(root, new_block_info):
    for name, index_set, children in new_block_info:
        new_block = Block(index_set)
        root.add_component(name, new_block)
        for data in new_block.values():
            for child in children:
                add_blocks(new_block, child)
