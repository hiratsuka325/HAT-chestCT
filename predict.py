import numpy as np
import tempfile
import shutil
import os
from PIL import Image
import subprocess
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def predict(
        self,
        image: Path = Input(
            description="Input Image.",
        ),
    ) -> Path:
        # 一時ディレクトリを作成
        input_dir = "input_dir"
        output_path = Path(tempfile.mkdtemp()) / "output.png"

        try:
            for d in [input_dir, "results"]:
                if os.path.exists(input_dir):
                    shutil.rmtree(input_dir)
            os.makedirs(input_dir, exist_ok=False)

            # 入力画像を入力ディレクトリにコピー
            input_path = os.path.join(input_dir, os.path.basename(image))
            shutil.copy(str(image), input_path)

            # 外部スクリプトを実行
            subprocess.call(
                [
                    "python",
                    "hat/test.py",
                    "-opt",
                    "options/test/HAT_SRx4_ImageNet-LR.yml",
                ]
            )

            # 結果ディレクトリのパスを設定
            res_dir = os.path.join(
                "results", "HAT_SRx4_ImageNet-LR", "visualization", "custom"
            )

            assert (
                len(os.listdir(res_dir)) == 1
            ), "Should contain only one result for Single prediction."

            # 結果画像を保存
            res = Image.open(os.path.join(res_dir, os.listdir(res_dir)[0]))
            res.save(str(output_path))

        finally:
            pass
            # 作成したディレクトリを削除
            shutil.rmtree(input_dir)
            shutil.rmtree("results")

        return output_path
