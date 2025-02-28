# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py  # noqa: E501
from abc import ABCMeta, abstractmethod


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class MemcachedBackend(BaseStorageBackend):
    """Memcached storage backend.

    Attributes:
        server_list_cfg (str): Config file for memcached server list.
        client_cfg (str): Config file for memcached client.
        sys_path (str | None): Additional path to be appended to `sys.path`.
            Default: None.
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError('Please install memcached to enable MemcachedBackend.')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(self.server_list_cfg, self.client_cfg)
        # mc.pyvector servers as a point which points to a memory cache
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf

    #def get_mhd(self, filepath):
        """MHDファイルを読み込み、画像データをバイト配列として返すメソッド"""
        import os

        filepath = str(filepath)
    
        try:
            with open(filepath, 'r') as f:
                # ヘッダー情報を取得
                header = {}
                header_lines = [line for line in f if line and not line.startswith('#')]
                header = {key.strip(): value.strip() for line in header_lines for key, value in [line.split('=')]}

            # 画像データのファイルパスを取得
            data_file = os.path.join(os.path.dirname(filepath), header['ElementDataFile'])

            # バイトデータを読み込む
            with open(data_file, 'rb') as f:
                byte_data = f.read()  # バイト配列を読み込む

            sizes = list(map(int, header['DimSize'].split()))
            return byte_data, sizes

        except FileNotFoundError as e:
            raise FileNotFoundError(f'MHD file {filepath} not found.') from e


class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
        db_paths (str | list[str]): Lmdb database paths.
        client_keys (str | list[str]): Lmdb client keys. Default: 'default'.
        readonly (bool, optional): Lmdb environment parameter. If True,
            disallow any write operations. Default: True.
        lock (bool, optional): Lmdb environment parameter. If False, when
            concurrent access occurs, do not lock the database. Default: False.
        readahead (bool, optional): Lmdb environment parameter. If False,
            disable the OS filesystem readahead mechanism, which may improve
            random read performance when a database is larger than RAM.
            Default: False.

    Attributes:
        db_paths (list): Lmdb database path.
        _client (list): A list of several lmdb envs.
    """

    def __init__(self, db_paths, client_keys='default', readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')

        if isinstance(client_keys, str):
            client_keys = [client_keys]

        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]
        assert len(client_keys) == len(self.db_paths), ('client_keys and db_paths should have the same length, '
                                                        f'but received {len(client_keys)} and {len(self.db_paths)}.')

        self._client = {}
        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(path, readonly=readonly, lock=lock, readahead=readahead, **kwargs)

    def get(self, filepath, client_key):
        """Get values according to the filepath from one lmdb named client_key.

        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
            client_key (str): Used for distinguishing different lmdb envs.
        """
        filepath = str(filepath)
        assert client_key in self._client, (f'client_key {client_key} is not ' 'in lmdb clients.')
        client = self._client[client_key]
        with client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class FileClient(object):
    """A general file client to access files in different backend.

    The client loads a file or text in a specified backend from its path
    and return it as a binary file. it can also register other backend
    accessor with a given name and backend class.

    Attributes:
        backend (str): The storage backend type. Options are "disk",
            "memcached" and "lmdb".
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'disk': HardDiskBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(f'Backend {backend} is not supported. Currently supported ones'
                             f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        # client_key is used only for lmdb, where different fileclients have
        # different lmdb environments.
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)
    
    def get_mhd(self, filepath):
        return self.client.get_mhd(filepath)