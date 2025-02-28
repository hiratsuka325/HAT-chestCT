from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_metas_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_mask
from basicsr.utils import FileClient, imfrombytes, img2tensor, imfrombytes_mhd, read_mhd_to_numpy
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.scale = opt['scale']

        self.gt_folder, self.lq_folder , self.mask_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_mask']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            #self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #                                              self.opt['meta_info_file'], self.filename_tmpl)
            self.paths = paired_metas_from_meta_info_file([self.lq_folder, self.gt_folder, self.mask_folder], ['lq', 'gt', 'mask'],
                                                          self.opt['meta_info_file'], self.filename_tmpl, self.scale)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        # getitemで画像を持ってくるとtakeへ何回もアクセスすることになるのでinitで画像を持ってくることで一回だけにするためにこの先を追加した
        import os
        import numpy as np
        self.lq_imgs = []
        self.gt_imgs = []
        self.mask_imgs = []
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        for meta in self.paths:
            # 入力画像（低解像度画像）
            lq_path = meta['lq_path']
            if os.path.exists(lq_path):
                lq_img = read_mhd_to_numpy(lq_path, float32=True) # (H, W, 1)
                self.lq_imgs.append(lq_img)
            else:
                print(f"Warning: {lq_path} does not exist.")
        
            # グラウンドトゥルース画像
            gt_path = meta['gt_path']
            if os.path.exists(gt_path):
                gt_img = read_mhd_to_numpy(gt_path, float32=True) # (H, W, 1)
                self.gt_imgs.append(gt_img)
            else:
                print(f"Warning: {gt_path} does not exist.")

            # マスク
            mask_path = meta['mask_path']
            if os.path.exists(mask_path):
                img_bytes = self.file_client.get(mask_path, 'mask')
                mask_img = imfrombytes(img_bytes, float32=True)

                # チャンネル次元を追加して(H, W, 1)にする
                if mask_img.ndim == 2:  # (H, W)の場合
                    mask_img = mask_img[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
                elif mask_img.ndim == 3 and mask_img.shape[2] != 1:  # (H, W, C) で C != 1の場合
                    mask_img = mask_img[:, :, :1]  # チャンネル数が多い場合、1チャンネルだけを取る
                    
                # 非ゼロ画素を1に変換
                mask_img = np.where(mask_img != 0, 1.0, 0.0)

                self.mask_imgs.append(mask_img)
            else:
                print(f"Warning: {mask_path} does not exist.")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.scale

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        #gt_size = self.paths[index]['gt_size']
        #img_bytes = self.file_client.get(gt_path, 'gt')
        #img_bytes, sizes = self.file_client.get_mhd(gt_path)
        #if img_bytes is None:
        #    print(f"Failed to load data from {gt_path}")
        #img_gt = imfrombytes(img_bytes, float32=True)
        #img_gt = imfrombytes_mhd(img_bytes, gt_size, float32=True)
        #img_gt = read_mhd_to_numpy(gt_path, float32=True)

        lq_path = self.paths[index]['lq_path']
        #lq_size = self.paths[index]['lq_size']
        #img_bytes = self.file_client.get(lq_path, 'lq')
        #img_bytes, sizes = self.file_client.get_mhd(lq_path)
        #img_lq = imfrombytes(img_bytes, float32=True)
        #img_lq = imfrombytes_mhd(img_bytes, lq_size, float32=True)
        #img_lq = read_mhd_to_numpy(lq_path, float32=True)

        mask_path = self.paths[index]['mask_path'] 
        #img_bytes = self.file_client.get(mask_path, 'mask')
        #img_mask = imfrombytes(img_bytes, float32=True)

        # getitemで画像を持ってくるとtakeへ何回もアクセスすることになるのでinitで画像を持ってくることで一回だけにするためにここを追加した
        img_gt = self.gt_imgs[index]
        img_lq = self.lq_imgs[index]
        img_mask = self.mask_imgs[index]
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            #img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq , img_mask = paired_random_crop_mask(img_gt, img_lq, img_mask, gt_size, scale, gt_path)
            # flip, rotation
            #img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt, img_lq , img_mask = augment([img_gt, img_lq, img_mask], self.opt['use_hflip'], self.opt['use_rot'])

            # なぜか(H, W, 1) -> (H, w)になるので追加した
        import numpy as np
        if img_gt.ndim == 2:  # (H, W)の場合
            img_gt = img_gt[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        if img_lq.ndim == 2:  # (H, W)の場合
            img_lq = img_lq[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        if img_mask.ndim == 2:  # (H, W)の場合
            img_mask = img_mask[:, :, np.newaxis]  # (H, W) -> (H, W, 1)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        #img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_gt, img_lq, img_mask = img2tensor([img_gt, img_lq, img_mask], bgr2rgb=False, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        #return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        #if self.opt['phase'] == 'train':
        #    return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        #else:
        #    return {'lq': img_lq, 'gt': img_gt, 'mask': img_mask, 'lq_path': lq_path, 'gt_path': gt_path, 'mask_path':mask_path}
        return {'lq': img_lq, 'gt': img_gt, 'mask': img_mask, 'lq_path': lq_path, 'gt_path': gt_path, 'mask_path':mask_path}

    def __len__(self):
        return len(self.paths)