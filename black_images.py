import os
from PIL import Image
import numpy as np

def is_black_image(image_path):
    """画像の要素が全て0か確認する関数"""
    try:
        # 画像を開く
        img = Image.open(image_path)
        
        # 画像をRGBに変換
        img = img.convert('RGB')
        
        # 画像データをnumpy配列に変換
        img_array = np.array(img)
        
        # すべてのピクセルが黒かどうかを確認
        if np.all(img_array == [0, 0, 0]):
            return True
        else:
            return False
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def check_images_in_directory(directory, output_file):
    """ディレクトリ内のすべての画像をチェックし、黒い画像をファイルに保存する関数"""
    black_images = []
    
    # ディレクトリ内のファイルを走査
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # 画像ファイルのみを対象にする（拡張子でフィルタリング）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            if is_black_image(file_path):
                black_images.append(filename)
    
    # 黒い画像のファイル名を指定したファイルに保存
    if black_images:
        with open(output_file, 'w') as f:
            for image in black_images:
                f.write(image + '\n')
        print(f"黒い画像のファイル名が '{output_file}' に保存されました。")
    else:
        print("黒い画像は見つかりませんでした。")

# 使用例
directory_path = "/path/to/your/directory"  # 対象ディレクトリのパスを指定
output_file = "black_images.txt"  # 保存するファイル名
check_images_in_directory(directory_path, output_file)