#!usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "AlexanderJustin"
# My blog: https://www.blog.csdn.net/AlexanderJustin183/article
__version__ = "3.0.1"

__doc__ = """
A module for Snake AI.
It can show game scores and average scores on matplotlib window.\
Date: 2022-11-22
"""

# % matplotlib wx  # If you are using IPython, uncomment this line.
import os

from matplotlib import pyplot as plt, use

# import numpy as np

# matplotlib: https://www.matplotlib.org
# if you have pip: use command `pip install matplotlib` to install.
# or you have conda: use command `CONDA.EXE install matplotlib``.

# you also need WxPython for this program.
# use ``pip install WXPYTHON`` to install.

SAVE = False

use("WxAgg")  # use WxPython GUI so the Pygame window will not change it's size.
# replace it with ``use("Qt5Agg")``, then you will see the Pygame window will be very small.
# It"s an odd problem, and I can"t fix it. (at least on my computer)
# TODO: fix the problem with PyQt and Pygame.
plt.ion()


def plot(scores, mean_scores, w):
    plt.clf()
    
    plt.title("Training...")
    mngr = plt.get_current_fig_manager()
    
    mngr.window.SetPosition((w, 0))
    plt.xlabel("# Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)  # , label="Score")
    plt.plot(mean_scores)  # , label="Mean Score")
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    # plt.legend()
    plt.show(block=False)


def save(final, cuda):
    if not os.path.exists("./figs"):
        os.makedirs("./figs")
    if not isinstance(final, float) and SAVE:
        plt.savefig("./figs/fig_%d_cuda=%s.jpg" % (final, cuda))
