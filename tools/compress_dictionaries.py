#!/usr/bin/env python3

import pathlib

import zstandard


def CompressDictionaries() -> None:
    """
    ../resources/dictionaries/ 以下の csv ファイルを ZStandard で圧縮し、
    同じディレクトリに .csv.zst として保存する
    """

    # 圧縮レベルは 1-22 まで指定可能
    # 数値が大きいほど圧縮率が高くなるが、圧縮・解凍に時間がかかる
    # 5 は圧縮率と解凍速度のバランスが良い値
    compression_level = 5

    # ../resources/dictionaries/ のパスを取得
    dictionaries_path = (
        pathlib.Path(__file__).parent.parent / "resources" / "dictionaries"
    )
    if not dictionaries_path.exists():
        print("Error: ../resources/dictionaries/ does not exist")
        return

    # csv ファイルを列挙
    csv_files = sorted(list(dictionaries_path.glob("**/*.csv")))
    if len(csv_files) == 0:
        print("Error: No csv files found")
        return

    # ZStandard の圧縮器を初期化
    compressor = zstandard.ZstdCompressor(level=compression_level)

    # csv ファイルを圧縮
    for csv_file in csv_files:
        # 出力先のパスを生成
        output_path = csv_file.with_suffix(".csv.zst")

        # 圧縮を実行
        with open(csv_file, "rb") as input_file:
            with open(output_path, "wb") as output_file:
                compressor.copy_stream(input_file, output_file)

        print(f"Compressed: {csv_file.name} -> {output_path.name}")


if __name__ == "__main__":
    CompressDictionaries()
