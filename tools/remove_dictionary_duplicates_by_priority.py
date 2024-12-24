"""
Usage: python remove_dictionary_duplicates_by_priority.py

resources/dictionaries ディレクトリにある辞書データのうち、ファイル名昇順で上位にある辞書の単語を優先し、
下位のファイルから重複する単語を削除するツール。
例えば 01_default.csv に含まれている単語が 04_neologd-01.csv にも存在する場合、
04_neologd-01.csv から該当する単語を削除する。

ただし、削除する側の単語コスト(3番目の値)が0以外で、残す側の単語コストが0の場合は、
残す側の単語コストを削除する側の値で上書きする (削除する側が neologd の場合のみ) 。
"""

import csv
import shutil
from pathlib import Path


def read_csv_words(csv_path: str) -> dict[tuple[str, str, str, str], list[str]]:
    """CSV ファイルから単語辞書を作成する"""
    words: dict[tuple[str, str, str, str], list[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # 空行をスキップ
                # 単語と品詞情報(4-6列目)をキーとして使用
                key = (row[0], row[4], row[5], row[6])
                words[key] = row
    return words


def process_auto_csv(
    file_path: str,
    priority_words: dict[tuple[str, str, str, str], list[str]],
    priority_file_path: str,
) -> tuple[int, list[tuple[list[str], list[str], bool]]]:
    """CSV を処理し、優先度の高いファイルに含まれる単語を削除する"""
    unique_rows: list[list[str]] = []
    removed_rows: list[tuple[list[str], list[str], bool]] = []

    # 優先ファイルの内容を更新可能な形式で読み込む
    priority_file_rows: list[list[str]] = []
    with open(priority_file_path, "r", encoding="utf-8") as pfile:
        priority_file_rows = list(csv.reader(pfile))

    with open(file_path, "r", encoding="utf-8") as auto_file:
        auto_reader = csv.reader(auto_file)
        for row in auto_reader:
            if row:
                # 単語と品詞情報(4-6列目)をキーとして使用
                key = (row[0], row[4], row[5], row[6])
                if key in priority_words:
                    priority_row = priority_words[key]
                    cost_updated = False

                    # 単語コストの比較と更新
                    # priority_file_path のファイル名に neologd が含まれている場合のみ実行する
                    if "neologd" in priority_file_path:
                        if row[3] != "0" and priority_row[3] == "0":
                            # 優先ファイル内の該当行を探して更新
                            for i, prow in enumerate(priority_file_rows):
                                if prow and (prow[0], prow[4], prow[5], prow[6]) == key:
                                    priority_file_rows[i][3] = row[3]
                                    priority_row = priority_file_rows[i]  # 更新後の行を使用
                                    cost_updated = True
                                    break

                    removed_rows.append((row, priority_row, cost_updated))
                else:
                    unique_rows.append(row)

    # 優先ファイルを更新された内容で上書き
    with open(priority_file_path, "w", encoding="utf-8", newline="") as pfile:
        writer = csv.writer(pfile, lineterminator="\n")
        writer.writerows(priority_file_rows)

    # 重複を削除した結果を書き込む
    with open(file_path, "w", encoding="utf-8", newline="") as auto_file:
        # 改行コードを LF にする
        writer = csv.writer(auto_file, lineterminator="\n")
        writer.writerows(unique_rows)

    # ファイルの末尾の改行を削除
    with open(priority_file_path, "rb+") as pfile:
        pfile.seek(-1, 2)
        last_char = pfile.read(1)
        if last_char == b"\n":
            pfile.seek(-1, 2)
            pfile.truncate()
    with open(file_path, "rb+") as auto_file:
        auto_file.seek(-1, 2)
        last_char = auto_file.read(1)
        if last_char == b"\n":
            auto_file.seek(-1, 2)
            auto_file.truncate()

    return len(unique_rows), removed_rows


def print_removed_entries(
    file_name: str,
    priority_file_name: str,
    removed_rows: list[tuple[list[str], list[str], bool]],
) -> None:
    """削除された行 (赤色) とそれに対応する優先ファイルの行 (緑色) を色付きで出力する"""
    print(f"\nRemoved entries from {file_name} (duplicates with {priority_file_name}):")
    print("-" * shutil.get_terminal_size().columns)
    for auto_row, priority_row, cost_updated in removed_rows:
        print(f'\033[91m{",".join(auto_row)}\033[0m')  # 赤色で削除された行を出力
        # コストが更新された場合は、その旨を表示
        if cost_updated:
            print(
                f'\033[92m{",".join(priority_row)} (cost updated from 0)\033[0m'
            )  # 緑色で優先ファイルの行を出力
        else:
            print(
                f'\033[92m{",".join(priority_row)}\033[0m'
            )  # 緑色で優先ファイルの行を出力
        print("-" * shutil.get_terminal_size().columns)


def remove_duplicates_by_priority() -> None:
    """ファイル名昇順で優先度を決定し、重複を削除する"""
    dict_dir = Path("resources/dictionaries")
    csv_files = sorted(dict_dir.glob("*.csv"))
    total_removed = 0

    # 各ファイルについて、それより後ろのファイルから重複を削除
    for i, priority_file in enumerate(csv_files):
        # 現在のファイルの内容を読み込む
        priority_words = read_csv_words(str(priority_file))

        # 現在のファイルより後ろのファイルを処理
        for target_file in csv_files[i + 1 :]:
            original_count = sum(1 for _ in open(target_file, "r", encoding="utf-8"))
            new_count, removed_rows = process_auto_csv(
                str(target_file), priority_words, str(priority_file)
            )
            removed = original_count - new_count
            total_removed += removed

            if removed > 0:
                print(f"Processed {target_file.name}: Removed {removed} entries")
                print_removed_entries(
                    target_file.name, priority_file.name, removed_rows
                )

    print(f"\nTotal removed entries: {total_removed}")


if __name__ == "__main__":
    remove_duplicates_by_priority()
