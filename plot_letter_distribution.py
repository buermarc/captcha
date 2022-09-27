import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import utils

import json

data = {}
with open("letter_accuracy.json", mode="r+") as _file:
    data = json.load(_file)

fig = plt.figure()
axs = fig.subplots(2, 1)

keys = [str(key) for key in data.keys()]
len_fh = int(np.ceil(len(keys)/2))
len_sh = int(len(keys) - len_fh)

fh_keys = keys[:len_fh]
sh_keys = keys[len_fh:]

fh_values = [data[fh_key] for fh_key in fh_keys]
sh_values = [data[sh_key] for sh_key in sh_keys]

all_values = fh_values.copy()
all_values.extend(sh_values)

fh_values = np.array(fh_values)
sh_values = np.array(sh_values)

_mean = np.mean(all_values)

mask1 = fh_values >= _mean
mask2 = fh_values < _mean

axs[0].bar(np.arange(len_fh)[mask1], fh_values[mask1], label=f">= {_mean:0.2f}", color="tab:blue")
axs[0].bar(np.arange(len_fh)[mask2], fh_values[mask2], label=f"< {_mean:0.2f}", color="tab:orange")
axs[0].set_xticks(np.arange(len_fh), fh_keys)
axs[0].legend(loc="lower left")

mask1 = sh_values >= _mean
mask2 = sh_values < _mean

axs[1].bar(np.arange(len_sh)[mask1], sh_values[mask1], color="tab:blue")
axs[1].bar(np.arange(len_sh)[mask2], sh_values[mask2], color="tab:orange")
axs[1].set_xticks(np.arange(len_sh), sh_keys)

# plt.tight_layout()
# plt.legend()
fig.suptitle(f"CVAE Classification Accuracy Per Class - Mean Value: {_mean:0.2f}")
#fig.suptitle(f"Mean Value: {_mean:0.2f}", pos="right")
#plt.title(f"Mean Value: {_mean:0.2f}", fontdict={"fontsize": 10})
# fig.subplots_adjust(hspace=0.4)
plt.savefig("barplot.pdf", bbox_inches="tight", dpi=200)
