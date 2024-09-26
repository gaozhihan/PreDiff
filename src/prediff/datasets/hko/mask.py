# Code is adapted from https://github.com/sxjscience/HKO-7/blob/master/nowcasting/mask.py
# File to deal read and write the .mask extensions
import zlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait

_executor_pool = ThreadPoolExecutor(max_workers=16)


def read_mask_file(filepath, out):
    """Load mask file to numpy array

    Parameters
    ----------
    filepath : str
    out : np.ndarray

    Returns
    -------

    """
    f = open(filepath, 'rb')
    dat = zlib.decompress(f.read())
    out[:] = np.frombuffer(dat, dtype=bool).reshape((480, 480))
    f.close()


def save_mask_file(npy_mask, filepath):
    compressed_data = zlib.compress(npy_mask.tobytes(), 2)
    f = open(filepath, "wb")
    f.write(compressed_data)
    f.close()


def quick_read_masks(path_list, layout="NCHW"):
    """
    Parameters
    ----------
    path_list:  list
    layout： str
        can be "NCHW", "NHWC", "NHW". "NHW" is only supported if grayscale is True.
        "NCHW" by default for compatibility.

    Returns
    -------

    """
    num = len(path_list)
    read_storage = np.empty((num, 480, 480), dtype=bool)
    # for i in range(num):
    #     read_storage[i] = read_mask_file(path_list[i])
    future_objs = []
    for i in range(num):
        obj = _executor_pool.submit(read_mask_file, path_list[i], read_storage[i])
        future_objs.append(obj)
    wait(future_objs)
    if layout == "NCHW":
        return read_storage.reshape((num, 1, 480, 480))
    elif layout == "NHW":
        return read_storage
    elif layout == "NHWC":
        return read_storage.reshape((num, 480, 480, 1))
    else:
        raise NotImplementedError
