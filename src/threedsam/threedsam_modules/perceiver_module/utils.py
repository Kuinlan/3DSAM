def sum_list(data, empty=None):
    """Sum a list of tensors or dictionaries"""
    if len(data) == 0:
        return empty
    output = data[0]
    for i in range(1, len(data)):
        if isinstance(output, list):
            output += data[i]
        elif isinstance(output, dict):
            for key in output.keys():
                output[key] += data[i][key]
        else:
            raise ValueError('Invalid sum')
    return output