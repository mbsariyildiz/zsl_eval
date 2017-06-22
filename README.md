Python implementation of zero-shot/generalized zero-shot evaluation framework proposed by Xian et al in Zero-Shot Learning - The Good, The Bad, The Ugly. You can find paper and source code at https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/.

Python script works on HDF5 compatible versions of dataset files provided by Yongqin Xian. Also some additional fields are required in the datasets, i.e. see evaluate.py. You can either download the datasets provided by Yongqin Xian and make them compatible with this work by using make_dataset.m (you can also modify any field as you wish, before using it consider relative path of dataset folder, see make_dataset.m) or you can just download the ones that are ready to use in this link:
https://drive.google.com/open?id=0Bx-ESazzDnp3TGpsY1VJa3dMRDA. 