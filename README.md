# MOOD HVAE
Code for team uOttawa's (me and Dr. Tanya Schmah) submission to the MOOD 2022
challenge.

Credits to the following codebases, which this implementation is built upon:
https://github.com/JakobHavtorn/hvae-oodd

https://github.com/MIC-DKFZ/mood

https://github.com/openai/pixel-cnn

All of the important source code is located in the `docker_example/scripts` directory.
The `docker_example/scripts/oodd` directory contains all of code defining the
HVAE model class and related layers & utilities (credit to Havtorn). The
`docker_example/scripts/pred_three_planes.py` file contains the code for evaluation
on folders of 3D images. You can write predictions out to a folder using any of
the top-level `.sh` scripts and following the same structure to adapt them to an
arbitrary dataset.
