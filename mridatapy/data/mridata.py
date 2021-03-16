#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import random
from typing import Dict, Optional, Union, Sequence, Tuple
import urllib
import zipfile

import ismrmrd
import numpy as np
import requests
from tqdm import tqdm


MRIDATA_ORG = 'http://mridata.org/'

# Totally, there are 20 cases among Stanford Fullysampled 3D FSE Knees.
# From http://mridata.org/list?project=Stanford%20Fullysampled%203D%20FSE%20Knees
UUIDS = [
    '52c2fd53-d233-4444-8bfd-7c454240d314',
    'b65b9167-a0d0-4295-9dd5-74641b3dd7e6',
    '8ad53ab7-07f9-4864-98d0-dc43145ff588',
    'cc70c22f-1ddc-4a53-9503-e33d613de321',
    '280cf3f9-3b7e-4738-84e0-f72b21aa5266',
    '38b9a8e8-2779-4979-8602-5e8e5f902863',
    '54c077b2-7d68-4e77-b729-16afbccae9ac',
    'ec00945c-ad90-46b7-8c38-a69e9e801074',
    'dd969854-ec56-4ccc-b7ac-ff4cd7735095',
    'efa383b6-9446-438a-9901-1fe951653dbd',
    '8eff1229-8074-41fa-8b5e-441b501f10e3',
    '7a9f028c-8667-48aa-8e08-0acf3320c8d4',
    'ee2efe48-1e9d-480e-9364-e53db01532d4',
    '9a740e7b-8fc3-46f9-9f70-1b7bedec37e4',
    '530a812a-4870-4d01-9db4-772c853d693c',
    '1b197efe-9865-43be-ac24-f237c380513e',
    '226e710b-725b-4bec-840e-bf47be2b8a44',
    '2588bfa8-0c97-478c-aa5a-487cc88a590d',
    'b7d435a1-2421-48d2-946c-d1b3372f7c60',
    'd089cbe0-48b3-4ae2-9475-53ca89ee90fe',
]

OLD_MRIDATA_ORG = 'http://old.mridata.org/'

# Chrome 89
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
]


class MRIData(object):
    """MRIData from Stanford Fullysampled 3D FSE Knees on mridata.org (or
    old.mridata.org).

    Totally 20 cases, for each case:
        |                     |               |
        | ------------------- | ------------- |
        | Number of Coils     | 8             |
        | Matrix Size         | 320 x 320 x 1 |
        | Number of Slices    | 256           |
        | Number of Phases    | 1             |
        | Number of Contrasts | 1             |
        | Trajectory          | Cartesian     |

    References:
        [1] Epperson, et al. Creation of Fully Sampled MR Data Repository for
        Compressed Sensing of the Knee. SMRT Conference, Salt Lake City, UT, 2013.
        [2] http://mridata.org/ (and http://old.mridata.org/)
    """
    _urls = []
    _filenames = []
    _data_type = 'ismrmrd'
    _data_dir = '.'

    def __init__(
        self,
        data_type: Optional[str] = None,
        path: Optional[Union[str, pathlib.Path]] = None
    ):
        """
        Args:
            data_type (Optional): Data type of mridata files that determines
                download URLs from either mridata.org or old.mridata.org.
                    - "ismrmrd" (Default): from mridata.org.
                    - "cfl": from old.mridata.org, can be loaded much faster.
            path (Optional): Directory where folder "mridata/" to be created.
                Default, "./mridata/".
        """
        data_type = data_type.lower() if data_type else self._data_type
        if data_type not in ('ismrmrd', 'cfl'):
            raise ValueError
        self._urls, self._filenames = self.get(data_type)
        self._data_type = data_type
        self._data_dir = pathlib.Path(path).joinpath('mridata')
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def urls(self):
        return self._urls

    @property
    def filenames(self):
        return self._filenames

    @property
    def type(self):
        return self._data_type

    @property
    def dir(self):
        return self._data_dir

    def download(
        self,
        num: Optional[int] = None
    ):
        """Downloads mridata of the given data type.

        Args:
            num (Optional): Number of data files to be downloaded. If not given,
                download all. `num` only works when not greater than 20. Totally,
                there are 20 cases.
        """
        downloaded = []
        for url, filename in zip(self._urls, self._filenames):
            if (num > 0) and (len(downloaded) == num):
                break
            self.fetch(url, filename, self._data_dir)
            downloaded.append(url)

    def to_np(
        self,
        stack: bool,
        num: Optional[int] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Loads mridata to complex-valued k-space NumPy arrays. If not exist,
        download first.

        Args:
            stack: Stack toggle, to determine if different cases are stacked.
            num (Optional): Number of data files to be loaded. If not given,
                load all existing "valid" files. `num` only works when not
                greater than the number of existing "valid" files. Maximum
                number of "valid" files should be 20.

        Returns:
            - Stacked: NumPy array of shape (Ncases, Nslices, Ncoils, Nx, Ny).
            - Otherwise: A dictionary mapping output names to corresponding
            NumPy arrays of shape (Nslices, Ncoils, Nx, Ny).
        """
        data = {}
        if self._data_type == 'ismrmrd':
            # only files "*.h5" can be recognized
            files = list(self._data_dir.glob('*.h5'))
            if not files:
                self.download(num=num)
                files = list(self._data_dir.glob('*.h5'))
            for file in files:
                if num and len(data) == num:
                    break
                a = self.ismrmrd_to_np(file)
                name = file.name.split('.')[0].lower()
                data[name] = a
        elif self._data_type == 'cfl':
            # either folders "p*"
            files = list(self._data_dir.glob('p*'))
            for file in files:
                if not file.is_dir():
                    files.remove(file)
            if not files:
                # or files "*.zip" can be recognized,
                # mixed types are not supported
                files = list(self._data_dir.glob('*.zip'))
                if not files:
                    self.download(num=num)
                    files = list(self._data_dir.glob('*.zip'))
            for file in files:
                if num and len(data) == num:
                    break
                a = self.cfl_to_np(file)
                name = file.name.split('.')[0].lower()
                data[name] = a
        if stack:
            return np.stack(list(data.values()), axis=0)
        else:
            return data

    def to_npy(
        self,
        stack: bool,
        path: Optional[Union[str, pathlib.Path]] = None,
        num: Optional[int] = None
    ):
        """Converts mridata to .npy files. If not exist, download first.

        Args:
            stack: Stack toggle, to determine if different cases are stacked.
                    - True: NumPy array of shape (Ncases, Nslices, Ncoils, Nx, Ny)
                    will be saved.
                    - False: NumPy array of shape (Nslices, Ncoils, Nx, Ny) for
                    each case will be individually saved.
            path (Optional): Output directory where .npy file to be saved. If
                not given, save to the same path as initialized, e.g.,
                "./mridata/".
            num (Optional): Number of data files to be converted. If not given,
                load all existing "valid" files. `num` only works when not
                greater than the number of existing "valid" files. Maximum
                number of "valid" files should be 20.
        """
        data = self.to_np(stack, num=num)
        path = pathlib.Path(path) if path else self._data_dir
        path.mkdir(parents=True, exist_ok=True)
        if stack:
            npyfile = path.joinpath('mridata.npy')
            np.save(npyfile, data)  # overwrite if exists
        else:
            for name, a in data.items():
                npyfile = path.joinpath(name + '.npy')
                np.save(npyfile, a)  # overwrite if exists

    @staticmethod
    def get(data_type: str) -> Tuple[Sequence[str], Sequence[str]]:
        """Gets whole lists of download URLs and filenames corresponding to the
        given data type of mridata files to be downloaded.

        Args:
            data_type: Data type of mridata files that determines download URLs
                from either mridata.org or old.mridata.org.
                    - "ismrmrd": from mridata.org.
                    - "cfl": from old.mridata.org, can be loaded much faster.

        Returns:
            Tuple of lists of download URLs and filenames.
        """
        data_type = data_type.lower()
        if data_type not in ('ismrmrd', 'cfl'):
            raise ValueError
        mridata_urls, mridata_filenames = [], []
        if data_type == 'ismrmrd':
            mridata_urls = [urllib.parse.urljoin(MRIDATA_ORG, 'download/%s'%(uuid)) for uuid in UUIDS]
            mridata_filenames = ['%s.h5'%(uuid) for uuid in UUIDS]
        elif data_type == 'cfl':
            mridata_urls = [urllib.parse.urljoin(OLD_MRIDATA_ORG, 'knees/fully_sampled/p%d/e1/s1/P%d.zip'%(i, i)) for i in range(1, 21)]
            mridata_filenames = ['P%d.zip'%(i) for i in range(1, 21)]
        return mridata_urls, mridata_filenames

    @staticmethod
    def fetch(
        url: str,
        filename: str,
        path: Union[str, pathlib.Path]
    ):
        """Fetches mridata given the specific download URL and filename.

        Args:
            url: URL that allows to download a mridata file.
            filename: Filename of output.
            path: Output directory where data file to be fetched.

        References:
            [1] https://github.com/mikgroup/mridata-python/blob/master/mridata/download.py
        """
        user_agent = random.choice(USER_AGENTS)
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get('Content-Length', 0))
        chunk_size = 1024
        total_chunks = (total_size + chunk_size - 1) // chunk_size
        file = pathlib.Path(path).joinpath(filename)
        with open(file, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=chunk_size),
                              total=total_chunks,
                              unit='KB'):
                if chunk:
                    f.write(chunk)

    @staticmethod
    def ismrmrd_to_np(
        file: Union[str, pathlib.Path],
        first_slice: Optional[bool] = None
    ) -> np.ndarray:
        """Loads .h5 ISMRMRD file to complex-valued k-space NumPy array.

        Args:
            file: Input ISMRMRD file, e.g., a "./mridata/<uuid>.h5".
            first_slice (Optional): If True, extract only the first slice of a
                k-space volumn (a case).

        Returns:
            NumPy array of shape (Nslices, Ncoils, Nx, Ny).

        References:
            [1] https://github.com/ismrmrd/ismrmrd-paper/blob/master/code/do_recon_python.py
        """
        file = pathlib.Path(file)
        if not file.is_file():
            raise ValueError
        dataset = ismrmrd.Dataset(file, create_if_needed=False)
        header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
        enc = header.encoding[0]

        # Dimensions
        num_slices = enc.encodingLimits.slice.maximum + 1
        num_coils = header.acquisitionSystemInformation.receiverChannels
        num_kx = enc.encodedSpace.matrixSize.x
        num_ky = enc.encodedSpace.matrixSize.y
        shape = (num_slices, num_coils, num_kx, num_ky)

        # Initialiaze a storage array
        a = np.zeros(shape, dtype=np.complex64)  # (Nslices, Ncoils, Nx, Ny)

        # Loop through the acquisitions, ignoring noise scans
        num_acq = dataset.number_of_acquisitions()
        if first_slice:
            num_acq = num_ky
        for i in tqdm(range(num_acq)):
            acq = dataset.read_acquisition(i)
            idx_slice = acq.idx.slice
            idx_ky = acq.idx.kspace_encode_step_1
            a[idx_slice, :, :, idx_ky] = acq.data

        return a

    @staticmethod
    def ismrmrd_to_npy(
        file: Union[str, pathlib.Path],
        path: Optional[Union[str, pathlib.Path]] = None,
        first_slice: Optional[bool] = None
    ):
        """Converts .h5 ISMRMRD file to .npy file.

        Args:
            file: Input ISMRMRD file, e.g., a "./mridata/<uuid>.h5".
            path: Output directory where .npy file to be saved. If not given,
                save under the same directory of input ISMRMRD file.
            first_slice (Optional): If True, extract only the first slice of a
                k-space volumn (a case) from the corresponding ISMRMRD file.
        """
        file = pathlib.Path(file)
        if not file.is_file():
            raise ValueError
        path = pathlib.Path(path) if path else file.parent
        path.mkdir(parents=True, exist_ok=True)
        npyfile = path.joinpath(file.name.split('.')[0].lower() + '.npy')
        if not npyfile.exists():
            a = MRIData.ismrmrd_to_np(file, first_slice=first_slice)
            np.save(npyfile, a)

    @staticmethod
    def cfl_to_np(file: Union[str, pathlib.Path]) -> np.ndarray:
        """Loads .cfl file to complex-valued k-space NumPy array.

        Args:
            file: Input data file, either a .zip file, e.g., "./mridata/P1.zip",
                or a folder, e.g., "./mridata/p1", that contains "kspace.hdr"
                and "kspace.cfl". If given a .zip file, original file will be
                deleted once unzipped.

        Returns:
            NumPy array of shape (Nslices, Ncoils, Nx, Ny).

        References:
            [1] https://github.com/mrirecon/bart/blob/master/python/cfl.py
        """
        file = pathlib.Path(file)
        folder = None
        if file.is_file():
            folder = file.parent.joinpath(file.name.split('.')[0].lower())
            MRIData.unzip(file, path=folder, remove=True)
        elif file.is_dir():
            folder = file
        else:
            raise ValueError  # file.exists() -> False
        if (not folder) and (not list(folder.glob('**/*.hdr'))):
            raise ValueError

        # get dims from .hdr
        hdrfile = list(folder.glob('**/*.hdr'))[0]
        with open(hdrfile, 'r') as hdr:
            hdr.readline()
            l = hdr.readline()
        dims = [int(i) for i in l.split()]

        # remove singleton dimensions from the end
        n = np.prod(dims)
        dims_cumprod = np.cumprod(dims)
        shape = dims[:np.searchsorted(dims_cumprod, n) + 1]

        # load data from .cfl
        cflfile = list(folder.glob('**/*.cfl'))[0]
        with open(cflfile, 'r') as cfl:
            a = np.fromfile(cfl, dtype=np.complex64, count=n)

        # reshape into (Nslices, Ncoils, Nx, Ny)
        a = a.reshape(shape, order='F')  # (Nx, Ny, Nslices, Ncoils) from .hdr
        a = np.transpose(a, axes=(2, 3, 0, 1))  # (Nslices, Ncoils, Nx, Ny)

        return  a

    @staticmethod
    def cfl_to_npy(
        file: Union[str, pathlib.Path],
        path: Optional[Union[str, pathlib.Path]] = None
    ):
        """Converts .cfl file to .npy file.

        Args:
            file: Input data file, either a .zip file, e.g., "./mridata/P1.zip",
                or a folder, e.g., "./mridata/p1", that contains "kspace.hdr"
                and "kspace.cfl". If given a .zip file, original file will be
                deleted once unzipped.
            path: Output directory where .npy file to be saved. If not given,
                save under the same directory of input data file.
        """
        file = pathlib.Path(file)
        path = pathlib.Path(path) if path else file.parent
        path.mkdir(parents=True, exist_ok=True)
        npyfile = path.joinpath(file.name.split('.')[0].lower() + '.npy')
        if not npyfile.exists():
            a = MRIData.cfl_to_np(file)
            np.save(npyfile, a)

    @staticmethod
    def unzip(
        file: Union[str, pathlib.Path],
        path: Optional[Union[str, pathlib.Path]] = None,
        remove: Optional[bool] = None
    ):
        """Unzip .zip file.

        Args:
            file: Input .zip file, e.g., "./mridata/P1.zip".
            path (Optional): Output directory where .zip file to be unzipped.
                If not given, unzip under the same directory as the input .zip
                file.
            remove (Optional): Remove toggle, to determine if original .zip file
                will be deleted once unzipped.
        """
        file = pathlib.Path(file)
        if not file.is_file():
            raise ValueError
        folder = file.parent.joinpath(file.name.split('.')[0].lower())
        path = pathlib.Path(path) if path else folder
        path.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(file):
            with zipfile.ZipFile(file, 'r') as zip:
                zip.extractall(path)
            if remove:
                file.unlink()  # delete .zip file
        else:
            raise ValueError
