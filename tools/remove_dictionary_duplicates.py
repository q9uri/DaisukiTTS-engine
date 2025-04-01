"""
Usage: python remove_dictionary_duplicates.py

resources/dictionaries ディレクトリにある辞書データのうち、pyopenjtalk-plus デフォルト辞書 (naist-jdic.csv / unidic-csj.csv) に
既に定義されている単語を削除するツール。自動生成された辞書データ内の単語のうち、pyopenjtalk-plus デフォルト辞書に
既に定義されている単語を削除するユースケースでの利用を想定している。

remove_dictionary_duplicates_by_priority.py と異なり、デフォルト辞書は基本手動で作られた高精度な辞書という前提のもと、
品詞情報を無視して単語のみで重複しているかどうかを判定する（デフォルト辞書の方が品詞情報も含めて正確と考えられるため）。
"""  # noqa

import csv
import shutil
from pathlib import Path


def read_manual_words(manual_csv_paths: list[str]) -> dict[str, list[str]]:
    """手動生成された CSV から単語辞書を作成する"""
    manual_words: dict[str, list[str]] = {}
    for manual_csv_path in manual_csv_paths:
        with open(manual_csv_path, "r", encoding="utf-8") as manual_file:
            manual_reader = csv.reader(manual_file)
            for row in manual_reader:
                if row:  # 空行をスキップ
                    manual_words[row[0]] = row
    return manual_words


def process_auto_csv(
    file_path: str, manual_words: dict[str, list[str]]
) -> tuple[int, list[tuple[list[str], list[str]]]]:
    """自動生成された CSV を処理し、重複を削除する"""
    unique_rows: list[list[str]] = []
    removed_rows: list[tuple[list[str], list[str]]] = []
    with open(file_path, "r", encoding="utf-8") as auto_file:
        auto_reader = csv.reader(auto_file)
        for row in auto_reader:
            if row and row[0] in manual_words:
                removed_rows.append((row, manual_words[row[0]]))
            else:
                unique_rows.append(row)
    # 重複を削除した結果を書き込む
    with open(file_path, "w", encoding="utf-8", newline="") as auto_file:
        # 改行コードを LF にするのがポイント
        writer = csv.writer(auto_file, lineterminator="\n")
        writer.writerows(unique_rows)

    # ファイルの末尾の改行を削除
    with open(file_path, "rb+") as file:
        file.seek(-1, 2)
        last_char = file.read(1)
        if last_char == b"\n":
            file.seek(-1, 2)
            file.truncate()

    return len(unique_rows), removed_rows


def print_removed_entries(
    file_name: str, removed_rows: list[tuple[list[str], list[str]]]
) -> None:
    """削除された行 (赤色) とそれに対応する手動生成 CSV の行 (緑色) を色付きで出力する"""
    print(f"\nRemoved entries from {file_name}:")
    print("-" * shutil.get_terminal_size().columns)
    for auto_row, manual_row in removed_rows:
        print(f"\033[91m{','.join(auto_row)}\033[0m")  # 赤色で自動生成 CSV の行を出力
        print(f"\033[92m{','.join(manual_row)}\033[0m")  # 緑色で手動生成 CSV の行を出力
        print("-" * shutil.get_terminal_size().columns)


def remove_duplicates() -> None:
    # pyopenjtalk-plus のデフォルト辞書のパス
    manual_csv_paths = [
        str(
            (
                Path(__file__).parent.parent.parent
                / "pyopenjtalk-plus/pyopenjtalk/dictionary/naist-jdic.csv"
            ).resolve()
        ),  # fmt: skip # noqa
        str(
            (
                Path(__file__).parent.parent.parent
                / "pyopenjtalk-plus/pyopenjtalk/dictionary/unidic-csj.csv"
            ).resolve()
        ),  # fmt: skip # noqa
    ]
    manual_words = read_manual_words(manual_csv_paths)

    dict_dir = Path("resources/dictionaries")
    total_removed = 0

    for file_path in sorted(dict_dir.glob("*.csv")):
        # 01_default.csv は明示的に手動作成されたデフォルト辞書なのでスキップ
        if file_path.name == "01_default.csv":
            continue
        original_count = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
        new_count, removed_rows = process_auto_csv(str(file_path), manual_words)
        removed = original_count - new_count
        total_removed += removed
        print(f"Processed {file_path.name}: Removed {removed} entries")
        print_removed_entries(file_path.name, removed_rows)

    print(f"\nTotal removed entries: {total_removed}")


if __name__ == "__main__":
    remove_duplicates()
