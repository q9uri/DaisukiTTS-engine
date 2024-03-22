import copy
import glob
import json
import os
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

from fastapi import HTTPException

from voicevox_engine.library_manager import LibraryManager

aivm_manifest_name = "aivm_manifest.json"


class TestLibraryManager(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = TemporaryDirectory()
        self.tmp_dir_path = Path(self.tmp_dir.name)
        self.engine_name = "AivisSpeech Engine"
        self.library_manger = LibraryManager(
            self.tmp_dir_path,
            "1.0.0",
            "AivisSpeech",
            self.engine_name,
            "1b4a5014-d9fd-11ee-b97d-83c170a68ed3",
        )
        self.library_filename = Path("test/test.aivm")
        with open("test/aivm_manifest.json") as f:
            self.aivm_manifest = json.loads(f.read())
            self.library_uuid = self.aivm_manifest["uuid"]
        with ZipFile(self.library_filename, "w") as zf:
            speaker_infos = glob.glob("speaker_info/**", recursive=True)
            for info in speaker_infos:
                zf.write(info)
            zf.writestr(aivm_manifest_name, json.dumps(self.aivm_manifest))
        self.library_file = open(self.library_filename, "br")

        # 以下は Unused import エラーにしないための暫定的なもの
        assert copy  # type: ignore
        assert os  # type: ignore
        assert BytesIO  # type: ignore
        assert HTTPException  # type: ignore

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
        self.library_file.close()
        self.library_filename.unlink()

    # def create_aivm_without_manifest(self, filename: str) -> None:
    #     with (
    #         ZipFile(filename, "w") as zf_out,
    #         ZipFile(self.library_filename, "r") as zf_in,
    #     ):
    #         for file in zf_in.infolist():
    #             buffer = zf_in.read(file.filename)
    #             if file.filename != aivm_manifest_name:
    #                 zf_out.writestr(file, buffer)

    # def create_aivm_manifest(self, **kwargs):
    #     aivm_manifest = copy.deepcopy(self.aivm_manifest)
    #     return {**aivm_manifest, **kwargs}

    # def test_installed_libraries(self) -> None:
    #     self.assertEqual(self.library_manger.installed_libraries(), {})

    #     self.library_manger.install_library(
    #         self.library_uuid,
    #         self.library_file,
    #     )
    #     # 内容はdownloadable_library.jsonを元に生成されるので、内容は確認しない
    #     self.assertEqual(
    #         list(self.library_manger.installed_libraries().keys())[0], self.library_uuid
    #     )

    #     self.library_manger.uninstall_library(self.library_uuid)
    #     self.assertEqual(self.library_manger.installed_libraries(), {})

    # def test_install_library(self) -> None:
    #     # エンジンが把握していないライブラリのテスト
    #     invalid_uuid = "52398bd5-3cc3-406c-a159-dfec5ace4bab"
    #     with self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(invalid_uuid, self.library_file)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {invalid_uuid} が見つかりません。",
    #     )

    #     # 不正なZIPファイルのテスト
    #     with self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(self.library_uuid, BytesIO())
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"音声ライブラリ {self.library_uuid} は不正なファイルです。",
    #     )

    #     # aivm_manifestの存在確認のテスト
    #     invalid_aivm_name = "test/invalid.aivm"
    #     self.create_aivm_without_manifest(invalid_aivm_name)
    #     with open(invalid_aivm_name, "br") as f, self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(self.library_uuid, f)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {self.library_uuid} にaivm_manifest.jsonが存在しません。",
    #     )

    #     # aivm_manifestのパースのテスト
    #     # Duplicate name: 'aivm_manifest.json'とWarningを吐かれるので、毎回作り直す
    #     self.create_aivm_without_manifest(invalid_aivm_name)
    #     with ZipFile(invalid_aivm_name, "a") as zf:
    #         zf.writestr(aivm_manifest_name, "test")

    #     with open(invalid_aivm_name, "br") as f, self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(self.library_uuid, f)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {self.library_uuid} のaivm_manifest.jsonは不正です。",
    #     )

    #     # aivm_manifestのパースのテスト
    #     invalid_aivm_manifest = self.create_aivm_manifest(version=10)
    #     self.create_aivm_without_manifest(invalid_aivm_name)
    #     with ZipFile(invalid_aivm_name, "a") as zf:
    #         zf.writestr(aivm_manifest_name, json.dumps(invalid_aivm_manifest))

    #     with open(invalid_aivm_name, "br") as f, self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(self.library_uuid, f)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {self.library_uuid} のaivm_manifest.jsonに不正なデータが含まれています。",
    #     )

    #     # aivm_manifestの不正なversionのテスト
    #     invalid_aivm_manifest = self.create_aivm_manifest(version="10")
    #     self.create_aivm_without_manifest(invalid_aivm_name)
    #     with ZipFile(invalid_aivm_name, "a") as zf:
    #         zf.writestr(aivm_manifest_name, json.dumps(invalid_aivm_manifest))

    #     with open(invalid_aivm_name, "br") as f, self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(self.library_uuid, f)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {self.library_uuid} のversionが不正です。",
    #     )

    #     # aivm_manifestの不正なmanifest_versionのテスト
    #     invalid_aivm_manifest = self.create_aivm_manifest(manifest_version="10")
    #     self.create_aivm_without_manifest(invalid_aivm_name)
    #     with ZipFile(invalid_aivm_name, "a") as zf:
    #         zf.writestr(aivm_manifest_name, json.dumps(invalid_aivm_manifest))

    #     with open(invalid_aivm_name, "br") as f, self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(self.library_uuid, f)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {self.library_uuid} のmanifest_versionが不正です。",
    #     )

    #     # aivm_manifestの未対応のmanifest_versionのテスト
    #     invalid_aivm_manifest = self.create_aivm_manifest(
    #         manifest_version="999.999.999"
    #     )
    #     self.create_aivm_without_manifest(invalid_aivm_name)
    #     with ZipFile(invalid_aivm_name, "a") as zf:
    #         zf.writestr(aivm_manifest_name, json.dumps(invalid_aivm_manifest))

    #     with open(invalid_aivm_name, "br") as f, self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(self.library_uuid, f)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {self.library_uuid} は未対応です。",
    #     )

    #     # aivm_manifestのインストール先エンジンの検証のテスト
    #     invalid_aivm_manifest = self.create_aivm_manifest(
    #         engine_uuid="26f7823b-20c6-40c5-bf86-6dd5d9d45c18"
    #     )
    #     self.create_aivm_without_manifest(invalid_aivm_name)
    #     with ZipFile(invalid_aivm_name, "a") as zf:
    #         zf.writestr(aivm_manifest_name, json.dumps(invalid_aivm_manifest))

    #     with open(invalid_aivm_name, "br") as f, self.assertRaises(HTTPException) as e:
    #         self.library_manger.install_library(self.library_uuid, f)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {self.library_uuid} は{self.engine_name}向けではありません。",
    #     )

    #     # 正しいライブラリをインストールして問題が起きないか
    #     library_path = self.library_manger.install_library(
    #         self.library_uuid, self.library_file
    #     )
    #     self.assertEqual(self.tmp_dir_path / self.library_uuid, library_path)

    #     self.library_manger.uninstall_library(self.library_uuid)

    #     os.remove(invalid_aivm_name)

    # def test_uninstall_library(self) -> None:
    #     # TODO: アンインストール出来ないライブラリをテストできるようにしたい
    #     with self.assertRaises(HTTPException) as e:
    #         self.library_manger.uninstall_library(self.library_uuid)
    #     self.assertEqual(
    #         e.exception.detail,
    #         f"指定された音声ライブラリ {self.library_uuid} はインストールされていません。",
    #     )

    #     self.library_manger.install_library(self.library_uuid, self.library_file)
    #     self.library_manger.uninstall_library(self.library_uuid)
