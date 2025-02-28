import os
import torch
from torchvision import transforms
import numpy as np
import random
from PIL import Image

def load_image(image_path):
    """PNGファイルを読み込む関数。読み込んだ画像をテンソルに変換して返す。"""
    img = Image.open(image_path).convert('L')  
    transform = transforms.ToTensor()
    img_tensor = transform(img)  # テンソルに変換
    return img_tensor

def calculate_zero_ratio(images):
    """画像内のゼロ画素の割合を計算する関数"""
    import numpy as np
    if isinstance(images, list):
        total_zero_pixels = 0
        total_pixels = 0
        for img in images:
            if torch.is_tensor(img):
                total_zero_pixels += (img == 0).sum().item()  # ゼロ画素の数
                total_pixels += img.numel()  # 総画素数
            else:
                total_zero_pixels += np.sum(img == 0)  # NumPy配列の場合、ゼロ画素の数
                total_pixels += img.size  # NumPy配列の場合、総画素数
        return total_zero_pixels / total_pixels  # ゼロ画素の割合を返す
    else:
        # 単一の画像の場合
        if torch.is_tensor(images):
            total_zero_pixels = (images == 0).sum().item()  # ゼロ画素の数
            total_pixels = images.numel()  # 総画素数
        else:
            total_zero_pixels = np.sum(images == 0)  # NumPy配列の場合
            total_pixels = images.size  # NumPy配列の場合、総画素数
        return total_zero_pixels / total_pixels  # ゼロ画素の割合を返す

def random_crop_mask(img_masks, patch_size):
    """マスク画像のランダムクロップを行う関数"""
    if isinstance(img_masks, list):
        img_masks = img_masks[0]  # 画像リストが渡された場合、最初の画像のみを使用
    
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_masks[0]) else 'Numpy'
        
    if input_type == 'Tensor':
        h, w = img_masks[0].size()[-2:]
    else:
        h, w = img_masks[0].shape[0:2]
    
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image size ({h}, {w}) is smaller than patch size ({patch_size}, {patch_size}).")
    
    # クロップ位置をランダムに選択
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    
    # クロップしたマスク画像
    cropped_mask = img_masks[top:top + patch_size, left:left + patch_size]
    if input_type == 'Tensor':
        cropped_mask = [v[top:top + patch_size, left:left + patch_size] for v in img_masks]
    else:
        cropped_mask = [v[top:top + patch_size, left:left + patch_size, ...] for v in img_masks]
        
    if len(cropped_mask) == 1:
        cropped_mask = cropped_mask[0]
    
    return cropped_mask

def save_images_with_no_crop(input_dir, output_file, patch_size):
    """ディレクトリ内のすべてのPNGマスク画像に対してランダムクロップを実行し、ゼロ画素率が高い場合の画像を保存する"""
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]  # PNGファイルをリストアップ

    no_crop_images = []

    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)

        img_mask = load_image(image_path)  # マスク画像を読み込む
        
        for _ in range(1000):
            cropped_mask = random_crop_mask(img_mask, patch_size)  # ランダムクロップ

            zero_ratio_mask = calculate_zero_ratio([cropped_mask])  # ゼロ画素率を計算

            if zero_ratio_mask < 0.20:  
                break            
        else:
            no_crop_images.append(image_name)

    # ファイルに結果を書き込む
    with open(output_file, 'w') as f:
        for image_name in no_crop_images:
            f.write(image_name + '\n')

# 使用例
input_dir = './datasets/ChestCT/mask/lung1'  # 画像が保存されているディレクトリのパス
output_file = './no_crop_images.txt'  # 結果を保存するファイルのパス
patch_size = 256  # クロップサイズ（パッチサイズ）

save_images_with_no_crop(input_dir, output_file, patch_size)