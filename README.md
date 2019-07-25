# Project Skeleton Utilities
This repository provides a skeleton pytorch codebase commonly used across my projects. Some examples projects using this are [factored3d](https://github.com/shubhtuls/factored3d), [CMR](https://github.com/akanazawa/cmr/), [3D-relnet](https://github.com/nileshkulkarni/relative3d), and [CSM](https://nileshkulkarni.github.io/csm/).

Feel free to modify and/or build on this.

## Usage Instructions
Any experiment using this code base typically creates a class that inherits from the ['Trainer' class in nnutils/train_utils.py](https://github.com/shubhtuls/project_skeleton/blob/master/nnutils/train_utils.py#L56). See [this file](https://github.com/akanazawa/cmr/blob/master/experiments/shape.py) in CMR code as an example. To define this 'Trainer', one has to simply instantiate the functions to define a dataset, model, register scalars to plot, and create visualizations. Once these are done, the training, logging, and saving models proceeds using the defined functions in the base class. Please see the flags in [nnutils/train_utils.py](https://github.com/shubhtuls/project_skeleton/blob/master/nnutils/train_utils.py) to see the default parameters.
