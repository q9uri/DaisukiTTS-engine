#!/usr/bin/env python3

import pathlib

import pyopenjtalk


def BuildDictionary() -> None:
    """
    ../resources/dictionaries/ 以下の csv ファイルを個別にコンパイルし、
    同じディレクトリに .dic ファイルとして出力する
    """

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

    # 各CSVファイルごとに個別にコンパイル
    for csv_file in csv_files:

        # 出力する .dic ファイルのパスを生成
        output_dic_path = csv_file.with_suffix('.dic')

        try:
            # pyopenjtalk 向けにビルド
            pyopenjtalk.mecab_dict_index(str(csv_file), str(output_dic_path))
            if not output_dic_path.is_file():
                raise RuntimeError(f"Failed to build dictionary: {csv_file.name}")

            print(f"Successfully built: {output_dic_path.name}")

        except Exception as e:
            print(f"Error while processing {csv_file.name}: {str(e)}")
            raise e


if __name__ == "__main__":
    BuildDictionary()
