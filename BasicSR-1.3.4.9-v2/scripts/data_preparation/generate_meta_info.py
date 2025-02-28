from os import path as osp
import os
from PIL import Image
from pathlib import Path
import SimpleITK as sitk

from basicsr.utils import scandir


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = 'datasets/DIV2K/DIV2K_train_HR_sub/'
    meta_info_txt = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')

def generate_meta_info_ImageNet():
    
    gt_folder = 'datasets/ImageNet/GT/'
    meta_info_txt = 'hat/data/meta_info/meta_info_ImageNet_GT.txt'

    dir_list = sorted(os.scandir(gt_folder), key=lambda entry: entry.name)


    print(f'Number of entries: {len(dir_list)}')
    
    for dir_path in dir_list:
        print(f'Name: {dir_path.name}')
        print(f'  Is file: {dir_path.is_file()}')
        print(f'  Is directory: {dir_path.is_dir()}')


    with open(meta_info_txt, 'w') as f:
        for dir_path in dir_list:

            img_list = sorted(list(scandir(dir_path)))

            for img_name in img_list:

                img_path = osp.join(dir_path.name, img_name)
                img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            
                width, height = img.size

                mode = img.mode
                if mode == 'RGB':
                    n_channel = 3
                elif mode == 'L':
                    n_channel = 1
                elif mode == 'CMYK':
                    n_channel = 4
                elif mode == 'RGBA':
                    n_channel =4
                else:
                    raise ValueError(f'Unsupported mode {mode}.')

                info = f'{img_path} ({height},{width},{n_channel})'
                print(info)
                f.write(f'{info}\n')

def generate_meta_info_mhd():
    """Generate meta info for MHD dataset."""
    
    mhd_folder = 'datasets/srtest/hr_0'
    meta_info_txt = 'hat/data/meta_info/meta_info_MHD.txt'

    # MHDファイルのリストを取得
    mhd_list = sorted([f for f in scandir(mhd_folder) if f.endswith('.mhd')])

    with open(meta_info_txt, 'w') as f:
        for idx, mhd_path in enumerate(mhd_list):
            # MHDファイルを読み込み
            img = sitk.ReadImage(osp.join(mhd_folder, mhd_path))

            # 画像のサイズとチャンネル数を取得
            size = img.GetSize()  # (depth, height, width)
            n_channel = img.GetNumberOfComponentsPerPixel()  # チャンネル数

            # 高さと幅を取得
            height, width = size  # 2D画像の場合、heightとwidthを取得

            info = f'{mhd_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')

def generate_meta_info_ChestCT():
    
    gt_folder = 'datasets/ChestCT/HR/'
    meta_info_txt_train = 'hat/data/meta_info/meta_info_ChestCT_1_3_2_train.txt'
    meta_info_txt_val = 'hat/data/meta_info/meta_info_ChestCT_1_3_2_val_part.txt'
    meta_info_txt_test = 'hat/data/meta_info/meta_info_ChestCT_1_3_2_test.txt'

    dir_list = sorted(os.scandir(gt_folder), key=lambda entry: entry.name)

    print(f'Number of entries: {len(dir_list)}')

    with open(meta_info_txt_train, 'w') as f:
        for dir_path in dir_list:

            if dir_path.name == 'lung1':
                img_list = sorted(list(scandir(dir_path)))

                for img_name in img_list:
                    if img_name.endswith(".mhd"):
                        img_path = osp.join(dir_path.name, img_name)
                        img = sitk.ReadImage(osp.join(gt_folder, img_path))

                        height, width = img.GetSize()
                        n_channel = 1

                        info = f'{img_path} ({height},{width},{n_channel})'
                        print(info)
                        f.write(f'{info}\n')

            #if dir_path.name == 'lung3':
            #    img_list = sorted(list(scandir(dir_path)))
#
            #    for img_name in img_list:
            #        if img_name.endswith(".mhd"):
            #            img_path = osp.join(dir_path.name, img_name)
            #            img = sitk.ReadImage(osp.join(gt_folder, img_path))

            #            height, width = img.GetSize()
            #            n_channel = 1

            #            info = f'{img_path} ({height},{width},{n_channel})'
            #            print(info)
            #            f.write(f'{info}\n')

    with open(meta_info_txt_val, 'w') as f:
        for dir_path in dir_list:

            if dir_path.name == 'lung3-200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750':
                img_list = sorted(list(scandir(dir_path)))

                for img_name in img_list:
                    if img_name.endswith(".mhd"):
                        img_path = osp.join(dir_path.name, img_name)
                        img = sitk.ReadImage(osp.join(gt_folder, img_path))

                        height, width = img.GetSize()
                        n_channel = 1

                        info = f'{img_path} ({height},{width},{n_channel})'
                        print(info)
                        f.write(f'{info}\n')

    with open(meta_info_txt_test, 'w') as f:
        for dir_path in dir_list:

            if dir_path.name == 'lung2-200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750':
                img_list = sorted(list(scandir(dir_path)))

                for img_name in img_list:
                    if img_name.endswith(".mhd"):
                        img_path = osp.join(dir_path.name, img_name)
                        img = sitk.ReadImage(osp.join(gt_folder, img_path))

                        height, width = img.GetSize()
                        n_channel = 1

                        info = f'{img_path} ({height},{width},{n_channel})'
                        print(info)
                        f.write(f'{info}\n')

if __name__ == '__main__':
    #generate_meta_info_div2k()
    #generate_meta_info_ImageNet()
    #generate_meta_info_mhd()
    generate_meta_info_ChestCT()