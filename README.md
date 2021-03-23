# mridataPy

mridataPy is

* a lightweight toolbox for downloading and processing mridata from mridata.org



mridataPy supports to

* download dataset from either [mridata.org](http://mridata.org/) or [old.mridata.org](http://old.mridata.org/)
* load mridata to NumPy arrays, which can be stored to .npy files
* generate sampling masks that can densely sample the center region in k-space while subsample the outer region based on acceleration factor
* provide the ground truth reconstructed by applying RSS coil-combination to fullysampled data
* evaluate reconstructions with MSE, NMSE, PSNR, SSIM metrics



## Quickstart

Download one case of [Stanford Fullysampled 3D FSE Knees](http://mridata.org/list?project=Stanford%20Fullysampled%203D%20FSE%20Knees) (totally, 20 cases) from [mridata.org](http://mridata.org/):

```python
import mridatapy

mridata = mridatapy.data.MRIData()
mridata.download(num=1)
```



### Dependencies and Installation

#### Package Dependencies

`pip` will handle all package dependencies.



#### Install mridataPy

```bash
$ pip install mridatapy
```



## Documentation

### Module `data`

#### `MRIData`

```python
class mridatapy.data.MRIData(data_type=None, path=None)
```

* `urls`

  **Attribute**: Whole lists of download URLs corresponding to mridata of the given data type.

* `filenames`

  **Attribute**: Whole lists of download filenames corresponding to mridata of the given data type.

* `type`

  **Attribute**: Data type of mridata.

* `dir`

  **Attribute**: Directory to the folder "mridata/" as the default path for mridata.

* `download(num=None)`

  **Instance method**: Downloads mridata of the given data type.

* `to_np(num=None, stack=None)`

  **Instance method**: Loads mridata to complex-valued k-space NumPy arrays. If not exist, download first.

* `to_npy(path=None, num=None, stack=None)`

  **Instance method**: Converts mridata to .npy files. If not exist, download first.

* `get(data_type)`

  **Static method**: Gets whole lists of download URLs and filenames corresponding to mridata of the given data type to be downloaded.

* `fetch(url, filename, path)`

  **Static method**: Fetches mridata given the specific pair of download URL and filename.

* `ismrmrd_to_np(file, filter=None, prewhiten=None, first_slice=None)`

  **Static method**: Loads .h5 ISMRMRD file to complex-valued k-space NumPy array.

* `ismrmrd_to_npy(file, path=None, filter=None, prewhiten=None, first_slice=None)`

  **Static method**: Converts .h5 ISMRMRD file to .npy file.

* `cfl_to_np(file)`

  **Static method**: Loads .cfl file to complex-valued k-space NumPy array.

* `cfl_to_npy(file, path=None)`

  **Static method**: Converts .cfl file to .npy file.

* `unzip(file, path=None, remove=None)`

  **Static method**: Unzips .zip file.

* `load_npy(file)`

  **Static method**: Loads .npy file.



#### `RandomLine`

```python
class mridatapy.data.RandomLine(acceleration_factor, center_fraction)
```

Generates a sampling mask of the given shape that can densely sample the center region in k-space while subsample the outer region based on acceleration factor. The mask randomly selects a subset of columns from input k-space data.

* `__call__(shape, dtype=numpy.complex64, max_attempts=30, tolerance=0.1, seed=None)`

  Magic method enables instances to behave like functions.



#### `EquispacedLine`

```python
class mridatapy.data.EquispacedLine(acceleration_factor, center_fraction)
```

Generates a sampling mask of the given shape that can densely sample the center region in k-space while subsample the outer region based on acceleration factor. The mask selects a roughly equispaced subset of columns from input k-space data.

* `__call__(shape, dtype=numpy.complex64, max_attempts=30, tolerance=0.1, seed=None)`

  Magic method enables instances to behave like functions.



#### `PoissonDisk`

```python
class mridatapy.data.PoissonDisk(acceleration_factor, center_fraction)
```

Generates a sampling mask of the given shape that can densely sample the center region in k-space while subsample the outer region based on acceleration factor. The mask selects a subset of points from input k-space data, characterized by the Poisson disk sampling pattern.

* `__call__(shape, dtype=numpy.complex64, max_attempts=30, tolerance=0.1, seed=None)`

  Magic method enables instances to behave like functions.



### Module `utils`

#### `fft_centered`

```python
function mridatapy.utils.fft_centered(input, shape=None, dim=None, norm=None)
```

Computes the centered N dimensional discrete Fourier transform (FFT) of input.



#### `ifft_centered`

```python
function mridatapy.utils.ifft_centered(input, shape=None, dim=None, norm=None)
```

Computes the centered N dimensional inverse discrete Fourier transform (IFFT) of input.



#### `root_sum_squares`

```python
function mridatapy.utils.root_sum_squares(input, dim, complex=False)
```

Computes the Root Sum of Squares (RSS) of input along the a given dimension (coil dimension).



### Module `metrics`

#### `mean_squared_error`

```python
function mridatapy.metrics.mean_squared_error(gt, pred)
```

Computes the Mean Squared Error (MSE) between two images.



#### `normalized_mse`

```python
function mridatapy.metrics.normalized_mse(gt, pred)
```

Computes the Normalized Mean Squared Error (NMSE) between two images.



#### `peak_signal_noise_ratio`

```python
function mridatapy.metrics.peak_signal_noise_ratio(gt, pred, data_range=None)
```

Computes the Peak Signal to Noise Ratio (PSNR) between two images.



#### `structural_similarity`

```python
function mridatapy.metrics.structural_similarity(gt, pred, data_range=None)
```

Computes the Structural Similarity Index (SSIM) between two images.



## Related Projects

* [mridata.org](http://mridata.org/)
* [mridata-python](https://github.com/mikgroup/mridata-python) ([pypi.org/project/mridata/](https://pypi.org/project/mridata/))
* [SigPy](https://github.com/mikgroup/sigpy)
