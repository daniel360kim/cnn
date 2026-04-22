import os
import sys


def set_visible_gpus(argv=None):
    argv = argv or sys.argv
    for i, arg in enumerate(argv):
        if arg.startswith("--gpus="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return
        if arg == "--gpus" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
            return
