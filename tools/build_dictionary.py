#!/usr/bin/env python3

import pathlib

import pyopenjtalk


def BuildDictionary() -> None:
    """
    ../resources/dictionaries/ 以下の csv ファイルを名前順で連結し、
    ビルド済みの user.dic を同じディレクトリに出力する
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

    # 一時ファイルのパスを生成
    tmp_csv_path = dictionaries_path / "tmp_dict.csv"
    output_dic_path = dictionaries_path / "default.dic"

    try:
        # 全ての csv ファイルを連結
        csv_text = ""
        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.endswith("\n"):
                    content += "\n"
                csv_text += content
            print(f"Concatenated: {csv_file.name}")

        # 連結した csv を一時ファイルに保存
        tmp_csv_path.write_text(csv_text, encoding="utf-8")

        # pyopenjtalk 向けにビルド
        pyopenjtalk.mecab_dict_index(str(tmp_csv_path), str(output_dic_path))
        if not output_dic_path.is_file():
            raise RuntimeError("Failed to build dictionary.")

        print(f"Successfully built: {output_dic_path.name}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

    finally:
        # 一時ファイルを削除
        if tmp_csv_path.exists():
            tmp_csv_path.unlink()


if __name__ == "__main__":
    BuildDictionary()
