import re

import torch


class globalNormalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = sample / torch.max(sample) * 255
        return (sample - torch.mean(sample)) / torch.std(sample)


def read_mhd(filename):
    data = {}
    with open(filename, "r") as f:
        for line in f:
            s = re.search("([a-zA-Z]*) = (.*)", line)
            data[s[1]] = s[2]

            if " " in data[s[1]]:
                data[s[1]] = data[s[1]].split(" ")
                for i in range(len(data[s[1]])):
                    if data[s[1]][i].replace(".", "").replace("-", "").isnumeric():
                        if "." in data[s[1]][i]:
                            data[s[1]][i] = float(data[s[1]][i])
                        else:
                            data[s[1]][i] = int(data[s[1]][i])
            else:
                if data[s[1]].replace(".", "").replace("-", "").isnumeric():
                    if "." in data[s[1]]:
                        data[s[1]] = float(data[s[1]])
                    else:
                        data[s[1]] = int(data[s[1]])
    return data
