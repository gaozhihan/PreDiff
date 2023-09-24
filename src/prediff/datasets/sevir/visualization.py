import os
from typing import Optional, Sequence, Union, Dict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from .sevir_cmap import get_cmap, VIL_COLORS, VIL_LEVELS


HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255

THRESHOLDS = (0, 16, 74, 133, 160, 181, 219, 255)

def plot_hit_miss_fa(ax, y_true, y_pred, thres):
    mask = np.zeros_like(y_true)
    mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
    mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
    mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
    mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
    cmap = ListedColormap(HMF_COLORS)
    ax.imshow(mask, cmap=cmap)

def plot_hit_miss_fa_all_thresholds(ax, y_true, y_pred, **unused_kwargs):
    fig = np.zeros(y_true.shape)
    y_true_idx = np.searchsorted(THRESHOLDS, y_true)
    y_pred_idx = np.searchsorted(THRESHOLDS, y_pred)
    fig[y_true_idx == y_pred_idx] = 4
    fig[y_true_idx > y_pred_idx] = 3
    fig[y_true_idx < y_pred_idx] = 2
    # do not count results in these not challenging areas.
    fig[np.logical_and(y_true < THRESHOLDS[1], y_pred < THRESHOLDS[1])] = 1
    cmap = ListedColormap(HMF_COLORS)
    ax.imshow(fig, cmap=cmap)

def vis_sevir_seq(
        save_path,
        seq: Union[np.ndarray, Sequence[np.ndarray]],
        label: Union[str, Sequence[str]] = "pred",
        norm: Optional[Dict[str, float]] = None,
        interval_real_time: float = 10.0,  plot_stride=2,
        label_rotation=0,
        label_offset=(-0.06, 0.4),
        fs=10,):
    """
    Parameters
    ----------
    seq:    Union[np.ndarray, Sequence[np.ndarray]]
        shape = (T, H, W). Float value 0-1 after `norm`.
    label:  Union[str, Sequence[str]]
        label for each sequence.
    norm:   Union[str, Dict[str, float]]
        seq_show = seq * norm['scale'] + norm['shift']
    interval_real_time: float
        The minutes of each plot interval
    """
    fontproperties = FontProperties()
    fontproperties.set_family('serif')
    # font.set_name('Times New Roman')
    fontproperties.set_size(fs)
    # font.set_weight("bold")

    if isinstance(seq, Sequence):
        seq_list = [ele.astype(np.float32) for ele in seq]
        assert isinstance(label, Sequence) and len(label) == len(seq)
        label_list = label
        seq_len_list = [len(ele) for ele in seq]
    elif isinstance(seq, np.ndarray):
        seq_list = [seq.astype(np.float32), ]
        assert isinstance(label, str)
        label_list = [label, ]
        seq_len_list = [len(seq), ]
    else:
        raise NotImplementedError
    max_len = max(seq_len_list)

    if norm is None:
        norm = {'scale': 255,
                'shift': 0}
    nrows = len(seq_list)
    ncols = (max_len - 1) // plot_stride + 1
    fig, ax = plt.subplots(nrows=nrows,
                           ncols=ncols,
                           figsize=(3 * ncols, 3 * nrows))

    cmap_dict = lambda s: {'cmap': get_cmap(s, encoded=True)[0],
                           'norm': get_cmap(s, encoded=True)[1],
                           'vmin': get_cmap(s, encoded=True)[2],
                           'vmax': get_cmap(s, encoded=True)[3]}

    for i, (seq, label, seq_len) in enumerate(zip(seq_list, label_list, seq_len_list)):
        ax[i][0].set_ylabel(ylabel=label, fontproperties=fontproperties, rotation=label_rotation)
        ax[i][0].yaxis.set_label_coords(label_offset[0], label_offset[1])
        for j in range(0, max_len, plot_stride):
            if j < seq_len:
                x = seq[j] * norm['scale'] + norm['shift']
                ax[i][j // plot_stride].imshow(x, **cmap_dict('vil'))
                if i == len(seq_list) - 1 and i > 0:  # the last row which is not the `in_seq`.
                    ax[-1][j // plot_stride].set_title(f"Min {int(interval_real_time * (j + plot_stride))}",
                                                       y=-0.25, fontproperties=fontproperties)
            else:
                ax[i][j // plot_stride].axis('off')

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].xaxis.set_ticks([])
            ax[i][j].yaxis.set_ticks([])

    # Legend of thresholds
    num_thresh_legend = len(VIL_LEVELS) - 1
    legend_elements = [Patch(facecolor=VIL_COLORS[i],
                             label=f'{int(VIL_LEVELS[i - 1])}-{int(VIL_LEVELS[i])}')
                       for i in range(1, num_thresh_legend + 1)]
    ax[0][0].legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(-1.2, -0.),
                    borderaxespad=0, frameon=False, fontsize='10')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_path)
    plt.close(fig)
