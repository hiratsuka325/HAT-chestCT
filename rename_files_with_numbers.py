import os
import re

def rename_files_with_numbers(directory, base_name):
    # ディレクトリ内の全ファイルを取得
    files = os.listdir(directory)
    
    for filename in files:
        old_file = os.path.join(directory, filename)
        
        # ファイルかどうかをチェック
        if os.path.isfile(old_file):
            # ファイル名から数字を抽出
            numbers = re.findall(r'\d+', filename)
            
            # 数字が見つかった場合
            if numbers:
                new_number = int(numbers[0])  # 最初の数字を使用
                new_file = os.path.join(directory, f"{base_name}{new_number}.png")
                
                # ファイル名を変更
                os.rename(old_file, new_file)
                print(f"Renamed: {old_file} to {new_file}")

if __name__ == "__main__":
    # 使用例
    dir_path = "datasets/ChestCT/mask/lung4"  # 変更したいディレクトリのパス
    base_name = "denoise_2dslices_1802"  # 新しいファイル名のベース
    rename_files_with_numbers(dir_path, base_name)