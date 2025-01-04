"ユーザー辞書関連の処理"

import gc
import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID, uuid4

import pyopenjtalk
from pydantic import TypeAdapter

from ..logging import logger
from ..utility.path_utility import get_save_dir, resource_root
from .model import UserDictWord
from .user_dict_word import (
    SaveFormatUserDictWord,
    UserDictInputError,
    WordProperty,
    convert_from_save_format,
    convert_to_save_format,
    create_word,
    part_of_speech_data,
    priority2cost,
)

F = TypeVar("F", bound=Callable[..., Any])


def mutex_wrapper(lock: threading.Lock) -> Callable[[F], F]:
    def wrap(f: F) -> F:
        def func(*args: Any, **kw: Any) -> Any:
            lock.acquire()
            try:
                return f(*args, **kw)
            finally:
                lock.release()

        return func  # type: ignore

    return wrap


resource_dir = resource_root()
save_dir = get_save_dir()

if not save_dir.is_dir():
    save_dir.mkdir(parents=True)

# デフォルトのファイルパス
# ビルド済みデフォルトユーザー辞書ディレクトリのパス
DEFAULT_DICT_DIR_PATH = resource_dir / "dictionaries"
# ユーザー辞書ファイルのパス
_USER_DICT_PATH = save_dir / "user_dict.json"
# ビルド済み辞書ファイルのパス
_COMPILED_DICT_PATH = save_dir / "user.dic"


# 同時書き込みの制御
mutex_user_dict = threading.Lock()
mutex_openjtalk_dict = threading.Lock()


_save_format_dict_adapter = TypeAdapter(dict[str, SaveFormatUserDictWord])


class UserDictionary:
    """ユーザー辞書"""

    def __init__(
        self,
        default_dict_dir_path: Path = DEFAULT_DICT_DIR_PATH,
        user_dict_path: Path = _USER_DICT_PATH,
        compiled_dict_path: Path = _COMPILED_DICT_PATH,
    ) -> None:
        """
        Parameters
        ----------
        default_dict_dir_path : Path
            ビルド済みデフォルトユーザー辞書ディレクトリのパス
        user_dict_path : Path
            ユーザー辞書ファイルのパス
        compiled_dict_path : Path
            ビルド済み辞書ファイルのパス
        """
        self._default_dict_dir_path = default_dict_dir_path
        self._user_dict_path = user_dict_path
        self._compiled_dict_path = compiled_dict_path
        # pytest から実行されているかどうか
        self._is_pytest = "pytest" in sys.argv[0] or "py.test" in sys.argv[0]

        # 初回起動時などまだユーザー辞書 JSON が存在しない場合、辞書登録例として「担々麺」の辞書エントリを書き込む
        # pytest から実行されている場合は書き込まない
        if not self._user_dict_path.is_file() and not self._is_pytest:
            self._write_to_json({
                "dc94a187-9881-43c9-a9c1-cebbf774a96d": create_word(WordProperty(
                    surface="担々麺",
                    pronunciation="タンタンメン",
                    accent_type=3,
                ))
            })  # fmt: skip

        # サーバーの起動高速化のため、前回起動時にビルド済みのユーザー辞書データがあれば、そのまま pyopenjtalk に適用する
        if self._compiled_dict_path.is_file():
            try:
                pyopenjtalk.update_global_jtalk_with_user_dict(
                    str(self._compiled_dict_path.resolve(strict=True))
                )
                logger.info("Compiled user dictionary applied.")
            except Exception as ex:
                logger.error("Failed to apply compiled user dictionary.", exc_info=ex)

        # バックグラウンドで辞書更新を行う (辞書登録量によっては数秒を要する)
        threading.Thread(target=self.update_dict, daemon=True).start()

    @mutex_wrapper(mutex_user_dict)
    def _write_to_json(self, user_dict: dict[str, UserDictWord]) -> None:
        """ユーザー辞書データをファイルへ書き込む。"""
        save_format_user_dict: dict[str, SaveFormatUserDictWord] = {}
        for word_uuid, word in user_dict.items():
            save_format_word = convert_to_save_format(word)
            save_format_user_dict[word_uuid] = save_format_word
        user_dict_json = _save_format_dict_adapter.dump_json(save_format_user_dict)
        self._user_dict_path.write_bytes(user_dict_json)

    @mutex_wrapper(mutex_openjtalk_dict)
    def update_dict(self) -> None:
        """辞書を更新する。"""
        default_dict_dir_path = self._default_dict_dir_path
        compiled_dict_path = self._compiled_dict_path

        # pytest 実行時かつ Windows ではなぜか辞書更新時に MeCab の初期化に失敗するので、辞書更新自体を無効化する
        if self._is_pytest and sys.platform == "win32":
            return

        start_time = time.time()

        random_string = uuid4()
        tmp_csv_path = compiled_dict_path.with_suffix(
            f".dict_csv-{random_string}.tmp"
        )  # CSV 形式辞書データの一時保存ファイル
        tmp_compiled_path = compiled_dict_path.with_suffix(
            f".dict_compiled-{random_string}.tmp"
        )  # ビルド済み辞書データの一時保存ファイル

        try:
            # 辞書.csv を作成
            csv_text = ""

            # ユーザー辞書データの追加
            user_dict = self.read_dict()
            for word_uuid in user_dict:
                word = user_dict[word_uuid]
                csv_text += (
                    "{surface},{context_id},{context_id},{cost},{part_of_speech},"
                    + "{part_of_speech_detail_1},{part_of_speech_detail_2},"
                    + "{part_of_speech_detail_3},{inflectional_type},"
                    + "{inflectional_form},{stem},{yomi},{pronunciation},"
                    + "{accent_type}/{mora_count},{accent_associative_rule}\n"
                ).format(
                    surface=word.surface,
                    context_id=word.context_id,
                    cost=priority2cost(word.context_id, word.priority),
                    part_of_speech=word.part_of_speech,
                    part_of_speech_detail_1=word.part_of_speech_detail_1,
                    part_of_speech_detail_2=word.part_of_speech_detail_2,
                    part_of_speech_detail_3=word.part_of_speech_detail_3,
                    inflectional_type=word.inflectional_type,
                    inflectional_form=word.inflectional_form,
                    stem=word.stem,
                    yomi=word.yomi,
                    pronunciation=word.pronunciation,
                    accent_type=word.accent_type,
                    mora_count=word.mora_count,
                    accent_associative_rule=word.accent_associative_rule,
                )

            # この時点で csv_text が空文字列のとき、ユーザー辞書が空なため処理を終了する
            # ユーザー辞書 CSV が空の状態で継続すると pyopenjtalk.mecab_dict_index() で Segmentation Fault が発生する
            if not csv_text:
                logger.info("User dictionary is empty. Skipping dictionary update.")
                return

            # 辞書データを辞書.csv へ一時保存
            tmp_csv_path.write_text(csv_text, encoding="utf-8")

            # 辞書.csv を OpenJTalk 用にビルド
            pyopenjtalk.mecab_dict_index(str(tmp_csv_path), str(tmp_compiled_path))
            if not tmp_compiled_path.is_file():
                raise RuntimeError("辞書のビルド時にエラーが発生しました。")

            # ユーザー辞書の適用を解除した後、ユーザー辞書ファイルを置き換え
            pyopenjtalk.unset_user_dict()
            tmp_compiled_path.replace(compiled_dict_path)

            # デフォルトユーザー辞書ディレクトリにある *.dic ファイルを名前順に取得
            dict_files = sorted(list(default_dict_dir_path.glob("**/*.dic")))
            # ユーザー辞書ファイルのパスを追加
            dict_files.append(compiled_dict_path)

            # ユーザー辞書を pyopenjtalk に適用
            # デフォルトのユーザー辞書ファイルと、先ほどビルドした辞書ファイルの両方を指定する
            dict_paths = [str(p.resolve(strict=True)) for p in dict_files]
            if dict_paths:  # 辞書ファイルが1つ以上存在する場合のみ実行
                pyopenjtalk.update_global_jtalk_with_user_dict(dict_paths)

            logger.info(f"User dictionary updated. ({time.time() - start_time:.2f}s)")

        except Exception as e:
            logger.error(
                f"Failed to update user dictionary. ({time.time() - start_time:.2f}s)",
                exc_info=e,
            )
            raise e

        finally:
            # 後処理
            if tmp_csv_path.exists():
                tmp_csv_path.unlink()
            if tmp_compiled_path.exists():
                tmp_compiled_path.unlink()

            # 強制的にメモリを開放
            gc.collect()

    @mutex_wrapper(mutex_user_dict)
    def read_dict(self) -> dict[str, UserDictWord]:
        """ユーザー辞書を読み出す。"""
        # 指定ユーザー辞書が存在しない場合、空辞書を返す
        if not self._user_dict_path.is_file():
            return {}

        with self._user_dict_path.open(encoding="utf-8") as f:
            save_format_dict = _save_format_dict_adapter.validate_python(json.load(f))
            result: dict[str, UserDictWord] = {}
            for word_uuid, word in save_format_dict.items():
                result[str(UUID(word_uuid))] = convert_from_save_format(word)
        return result

    def import_user_dict(
        self, dict_data: dict[str, UserDictWord], override: bool = False
    ) -> None:
        """
        ユーザー辞書をインポートする。
        Parameters
        ----------
        dict_data : dict[str, UserDictWord]
            インポートするユーザー辞書のデータ
        override : bool
            重複したエントリがあった場合、上書きするかどうか
        """
        # インポートする辞書データのバリデーション
        for word_uuid, word in dict_data.items():
            UUID(word_uuid)
            for pos_detail in part_of_speech_data.values():
                if word.context_id == pos_detail.context_id:
                    assert word.part_of_speech == pos_detail.part_of_speech
                    assert (
                        word.part_of_speech_detail_1
                        == pos_detail.part_of_speech_detail_1
                    )
                    assert (
                        word.part_of_speech_detail_2
                        == pos_detail.part_of_speech_detail_2
                    )
                    assert (
                        word.part_of_speech_detail_3
                        == pos_detail.part_of_speech_detail_3
                    )
                    assert (
                        word.accent_associative_rule
                        in pos_detail.accent_associative_rules
                    )
                    break
            else:
                raise ValueError("対応していない品詞です")

        # 既存辞書の読み出し
        old_dict = self.read_dict()

        # 辞書データの更新
        # 重複エントリの上書き
        if override:
            new_dict = {**old_dict, **dict_data}
        # 重複エントリの保持
        else:
            new_dict = {**dict_data, **old_dict}

        # 更新された辞書データの保存と適用
        self._write_to_json(new_dict)
        self.update_dict()

    def apply_word(self, word_property: WordProperty) -> str:
        """新規単語を追加し、その単語に割り当てられた UUID を返す。"""
        # 新規単語の追加による辞書データの更新
        user_dict = self.read_dict()
        word_uuid = str(uuid4())
        user_dict[word_uuid] = create_word(word_property)

        # 更新された辞書データの保存と適用
        self._write_to_json(user_dict)
        self.update_dict()

        return word_uuid

    def rewrite_word(self, word_uuid: str, word_property: WordProperty) -> None:
        """単語 UUID で指定された単語を上書き更新する。"""
        # 既存単語の上書きによる辞書データの更新
        user_dict = self.read_dict()
        if word_uuid not in user_dict:
            raise UserDictInputError("UUID に該当する単語が見つかりませんでした")
        user_dict[word_uuid] = create_word(word_property)

        # 更新された辞書データの保存と適用
        self._write_to_json(user_dict)
        self.update_dict()

    def delete_word(self, word_uuid: str) -> None:
        """単語 UUID で指定された単語を削除する。"""
        # 既存単語の削除による辞書データの更新
        user_dict = self.read_dict()
        if word_uuid not in user_dict:
            raise UserDictInputError("UUID に該当する単語が見つかりませんでした")
        del user_dict[word_uuid]

        # 更新された辞書データの保存と適用
        self._write_to_json(user_dict)
        self.update_dict()
