"""
Usage: python remove_redundant_dictionary_entries.py

resources/dictionaries ディレクトリにある辞書データのうち、pyopenjtalk デフォルト辞書で得られる読みと一致する単語を削除するツール。
"""

import csv
import shutil
from pathlib import Path
from typing import cast

import pyopenjtalk


def get_default_reading(text: str) -> str:
    """pyopenjtalk のデフォルト辞書から読みを取得する"""
    # g2p() の kana=True で読みをカタカナで取得
    return cast(str, pyopenjtalk.g2p(text, kana=True))


def process_csv_file(file_path: str) -> tuple[int, list[list[str]]]:
    """CSV ファイルを処理し、デフォルト辞書の読みと一致する行を削除する"""
    unique_rows: list[list[str]] = []
    removed_rows: list[list[str]] = []
    total_rows = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
    processed_rows = 0

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            processed_rows += 1
            if processed_rows % 100 == 0:  # 100行ごとに進捗を表示
                print(f"Processing... {processed_rows}/{total_rows} rows", end="\r")

            if not row:  # 空行をスキップ
                continue

            # CSV の形式は 表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音 を想定
            surface = row[0]
            pronunciation = row[12].replace(
                ":", ""
            )  # 「読み」ではなく「発音」(一番最後) の方を採用する (: は除去してから比較)

            # デフォルト辞書から読みを取得
            default_reading = get_default_reading(surface)

            # デフォルト辞書の読みと完全一致する場合は削除
            if default_reading == pronunciation:
                removed_rows.append(row)
                print(
                    f'Removed: {surface} → {default_reading} | \033[91m{",".join(row)}\033[0m'
                )
            else:
                unique_rows.append(row)

    print(f"\nProcessed all {total_rows} rows")

    # 重複を削除した結果を書き込む
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(unique_rows)

    # ファイルの末尾の改行を削除
    with open(file_path, "rb+") as f:
        f.seek(-1, 2)
        last_char = f.read(1)
        if last_char == b"\n":
            f.seek(-1, 2)
            f.truncate()

    return len(unique_rows), removed_rows


def print_removed_entries(file_name: str, removed_rows: list[list[str]]) -> None:
    """削除された行を赤色で出力する"""
    print(f"\nSummary of removed entries from {file_name}:")
    print("-" * shutil.get_terminal_size().columns)
    for row in removed_rows:
        print(f'\033[91m{",".join(row)}\033[0m')  # 赤色で出力
        print("-" * shutil.get_terminal_size().columns)


def remove_redundant_dictionary_entries() -> None:
    dict_dir = Path("resources/dictionaries")
    total_removed = 0

    for file_path in sorted(dict_dir.glob("*.csv")):
        # 01_default.csv は手動生成された辞書なのでスキップ
        if file_path.name == "01_default.csv":
            continue

        print(f"\nProcessing {file_path.name}...")
        original_count = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
        new_count, removed_rows = process_csv_file(str(file_path))
        removed = original_count - new_count
        total_removed += removed

        print(f"Finished {file_path.name}: Removed {removed} entries")
        print_removed_entries(file_path.name, removed_rows)

    print(f"\nTotal removed entries: {total_removed}")


if __name__ == "__main__":
    remove_redundant_dictionary_entries()
