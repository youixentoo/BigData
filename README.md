# BigData
BigData assignment, Thijs Weenink.<br>
A simple classifier for the [PatchCamelyon](https://patchcamelyon.grand-challenge.org/) dataset.<br>
Don't forget to edit the filepath in `network.py` to point to the correct location of the datasets.

## Versions of modules used
tensorflow 2.10.1<br>
tensorflow-gpu 2.8.4<br>
scikit-learn 1.2.2<br>
matplotlib 3.6.0<br>
h5py 3.8.0

## Notes
From my testing it doesn't require more than 3.5GiB RAM when caching is off.<br>
It was using at most around, if not more than, 11GiB with it turned on.<br>
The 11GiB was spread among the GPU and CPU as my GPU only has 8 GB max.
