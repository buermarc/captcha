import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import utils

heatmap = torch.load("heatmap.pt").numpy()

_max = np.max(heatmap)

heatmap /= _max

_xticks = np.arange(62)
labels = utils.decode_label(_xticks + 1)

plt.figure(figsize=(16, 16))
a = plt.imshow(heatmap, cmap="magma", interpolation=None)
plt.xticks(_xticks, labels)
plt.yticks(_xticks, labels)
plt.xlabel("Predicted Class", fontsize=26)
plt.ylabel("Correct Class", fontsize=26)
a.axes.tick_params(labelsize=20)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.title("CVAE Classification Heatmap", fontdict={"fontsize": 36})
plt.savefig("heatmap.pdf", bbox_inches="tight", dpi=200)
