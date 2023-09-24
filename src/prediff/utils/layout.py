from copy import deepcopy
from typing import Dict, Any


def layout_to_in_out_slice(layout, in_len, out_len=None):
    t_axis = layout.find("T")
    num_axes = len(layout)
    in_slice = [slice(None, None), ] * num_axes
    out_slice = deepcopy(in_slice)
    in_slice[t_axis] = slice(None, in_len)
    if out_len is None:
        out_slice[t_axis] = slice(in_len, None)
    else:
        out_slice[t_axis] = slice(in_len, in_len + out_len)
    return in_slice, out_slice


def parse_layout_shape(layout: str) -> Dict[str, Any]:
    r"""

    Parameters
    ----------
    layout: str
            e.g., "NTHWC", "NHWC".

    Returns
    -------
    ret:    Dict
    """
    batch_axis = layout.find("N")
    t_axis = layout.find("T")
    h_axis = layout.find("H")
    w_axis = layout.find("W")
    c_axis = layout.find("C")
    return {
        "batch_axis": batch_axis,
        "t_axis": t_axis,
        "h_axis": h_axis,
        "w_axis": w_axis,
        "c_axis": c_axis,
    }
