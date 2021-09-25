import hashlib
import re
import os
import sys
import shutil
import tarfile
import subprocess
from tqdm import tqdm
from urllib.request import urlretrieve
from mindex_info import MINDEX_INFO


# https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


# For large files, we need to compute MD5 block by block. See:
# https://stackoverflow.com/questions/1131220/get-md5-hash-of-big-files-in-python
def compute_md5(file, block_size=2**20):
    m = hashlib.md5()
    with open(file, 'rb') as f:
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()


def get_cache_home():
    return os.path.expanduser(os.path.join(f'~{os.path.sep}.cache', "pya0"))


def download_url(url, dest, md5=None, force=False, verbose=True):
    if verbose:
        print(f'Downloading {url} to {dest}...')

    # Check to see if file already exists, if so, simply return (quietly) unless force=True, in which case we remove
    # destination file and download fresh copy.
    if os.path.exists(dest):
        if verbose:
            print(f'{dest} already exists!')
        if not force:
            if verbose:
                print(f'Skipping download.')
            return dest
        if verbose:
            print(f'force=True, removing {dest}; fetching fresh copy...')
        os.remove(dest)

    cookies = {}
    if url.find('drive.google.com') >= 0:
        import gdown
        gdown.download(url, dest, quiet=False)
    else:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
            urlretrieve(url, dest, reporthook=t.update_to)

    if md5:
        md5_computed = compute_md5(dest)
        assert md5_computed == md5, f'{dest} does not match checksum! Expecting {md5} got {md5_computed}.'

    return dest


def download_and_unpack_index(url, index_name, index_directory='indexes', force=False, verbose=True, prebuilt=False, md5=None):
    if prebuilt:
        index_directory = os.path.join(get_cache_home(), index_directory)
        index_path = os.path.join(index_directory, f'{index_name}.{md5}')

        if not os.path.exists(index_directory):
            os.makedirs(index_directory)

        local_tarball = os.path.join(index_directory, f'{index_name}.tar.gz')
        # If there's a local tarball, it's likely corrupted, because we remove the local tarball on success (below).
        # So, we want to remove.
        if os.path.exists(local_tarball):
            os.remove(local_tarball)
    else:
        local_tarball = os.path.join(index_directory, f'{index_name}.tar.gz')
        index_path = os.path.join(index_directory, f'{index_name}')

    # Check to see if index already exists, if so, simply return (quietly) unless force=True, in which case we remove
    # index and download fresh copy.
    if os.path.exists(index_path):
        if not force:
            if verbose:
                print(f'{index_path} already exists, skipping download.')
            return index_path
        if verbose:
            print(f'{index_path} already exists, but force=True, removing {index_path} and fetching fresh copy...')
        shutil.rmtree(index_path)

    dest = os.path.join(index_directory, f'{index_name}.tar.gz')
    download_url(url, dest, verbose=verbose, md5=md5)

    if verbose:
        print(f'Extracting {local_tarball} into {index_path}...')
    tarball = tarfile.open(local_tarball)
    tarball.extractall(index_directory)
    tarball.close()
    os.remove(local_tarball)

    if prebuilt:
        os.rename(os.path.join(index_directory, f'{index_name}'), index_path)

    return index_path


def download_prebuilt_index(index_name, force=False, verbose=True, mirror=None):
    if index_name not in MINDEX_INFO:
        raise ValueError(f'Unrecognized index name {index_name}')
    else:
        target_index = MINDEX_INFO[index_name]

    index_md5 = target_index['md5']
    for url in target_index['urls']:
        try:
            return download_and_unpack_index(url, index_name, prebuilt=True, md5=index_md5, verbose=verbose)
        except Exception as e:
            print(str(e), file=sys.stderr)
            print(f'Unable to download pre-built index at {url}, trying next URL...', file=sys.stderr)
    raise ValueError(f'Unable to download pre-built index at any known URLs.')


def mount_image_index(image_path, image_fs):
    mount_dir = os.path.dirname(image_path) + '/mnt-' + os.path.basename(image_path)
    os.makedirs(mount_dir, exist_ok=True)
    if not os.path.ismount(mount_dir):
        subprocess.run(["sudo", "umount", mount_dir])
        subprocess.run(["sudo", "mount", "-t", image_fs, image_path, mount_dir])
    return mount_dir
