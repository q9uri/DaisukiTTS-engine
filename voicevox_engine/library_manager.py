import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException
from pydantic import ValidationError
from semver.version import Version

from voicevox_engine.model import (
    AivmManifest,
    DownloadableLibraryInfo,
    InstalledLibraryInfo,
)

__all__ = ["LibraryManager"]

INFO_FILE = "metas.json"


class LibraryManager:
    """AIVM (Aivis Voice Model) ライブラリの管理"""

    def __init__(
        self,
        library_root_dir: Path,
        supported_aivm_version: str | None,
        brand_name: str,
        engine_name: str,
        engine_uuid: str,
    ):
        self.library_root_dir = library_root_dir
        self.library_root_dir.mkdir(exist_ok=True)
        if supported_aivm_version is not None:
            self.supported_aivm_version = Version.parse(supported_aivm_version)
        else:
            # supported_aivm_version が None の時は 0.0.0 として扱う
            self.supported_aivm_version = Version.parse("0.0.0")
        self.engine_brand_name = brand_name
        self.engine_name = engine_name
        self.engine_uuid = engine_uuid

    def downloadable_libraries(self) -> list[DownloadableLibraryInfo]:
        """
        ダウンロード可能ライブラリの一覧を取得する

        Returns
        -------
        - : list[DownloadableLibraryInfo]
        """

        # AivisSpeech Engine では実装しない
        return []

    def installed_libraries(self) -> dict[str, InstalledLibraryInfo]:
        """
        インストール済み AIVM ライブラリの情報を取得する

        Returns
        -------
        library : Dict[str, InstalledLibraryInfo]
            インストール済み AIVM ライブラリの情報
        """

        library: dict[str, InstalledLibraryInfo] = {}
        for library_dir in self.library_root_dir.iterdir():
            if library_dir.is_dir():
                # ライブラリ情報の取得 from `library_root_dir / f"{library_uuid}" / "metas.json"`
                library_uuid = os.path.basename(library_dir)
                with open(library_dir / INFO_FILE, encoding="utf-8") as f:
                    info = json.load(f)
                # アンインストール出来ないライブラリを作る場合、何かしらの条件でFalseを設定する
                library[library_uuid] = InstalledLibraryInfo(**info, uninstallable=True)

        return library

    def install_library(self, library_id: str, file: BinaryIO) -> Path:
        """
        AIVM ライブラリファイル (`.aivm`) をインストールする

        Parameters
        ----------
        library_id : str
            インストール対象のライブラリ ID
        file : BytesIO
            ライブラリファイルの Blob

        Returns
        -------
        library_dir : Path
            インストール済みライブラリの情報
        """

        for downloadable_library in self.downloadable_libraries():
            if downloadable_library.uuid == library_id:
                library_info = downloadable_library.model_dump()
                break
        else:
            raise HTTPException(
                status_code=404,
                detail=f"指定された音声ライブラリ {library_id} が見つかりません。",
            )

        # ライブラリディレクトリの生成
        library_dir = self.library_root_dir / library_id
        library_dir.mkdir(exist_ok=True)

        # metas.jsonの生成
        with open(library_dir / INFO_FILE, "w", encoding="utf-8") as f:
            json.dump(library_info, f, indent=4, ensure_ascii=False)

        # zipファイル形式のバリデーション
        if not zipfile.is_zipfile(file):
            raise HTTPException(
                status_code=422,
                detail=f"音声ライブラリ {library_id} は不正なファイルです。",
            )

        with zipfile.ZipFile(file) as zf:
            if zf.testzip() is not None:
                raise HTTPException(
                    status_code=422,
                    detail=f"音声ライブラリ {library_id} は不正なファイルです。",
                )

            # マニフェストファイルの存在とファイル形式をバリデーション
            aivm_manifest = None
            try:
                aivm_manifest = json.loads(
                    zf.read("aivm_manifest.json").decode("utf-8")
                )
            except KeyError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} に aivm_manifest.json が存在しません。",
                )
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} の aivm_manifest.json は不正です。",
                )

            # マニフェスト形式のバリデーション
            try:
                AivmManifest.model_validate(aivm_manifest)
            except ValidationError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} の aivm_manifest.json に不正なデータが含まれています。",
                )

            # ライブラリバージョンのバリデーション
            if not Version.is_valid(aivm_manifest["version"]):
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} の version が不正です。",
                )

            # マニフェストバージョンのバリデーション
            try:
                aivm_manifest_version = Version.parse(aivm_manifest["manifest_version"])
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} の manifest_version が不正です。",
                )
            if aivm_manifest_version > self.supported_aivm_version:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} は未対応です。",
                )

            # ライブラリ-エンジン対応のバリデーション
            if aivm_manifest["engine_uuid"] != self.engine_uuid:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {library_id} は {self.engine_name} 向けではありません。",
                )

            # 展開によるインストール
            zf.extractall(library_dir)

        return library_dir

    def uninstall_library(self, library_id: str) -> None:
        """
        インストール済み AIVM ライブラリをアンインストールする

        Parameters
        ----------
        library_id : str
            インストール対象のライブラリ ID
        """

        # 対象ライブラリがインストール済みであることの確認
        installed_libraries = self.installed_libraries()
        if library_id not in installed_libraries.keys():
            raise HTTPException(
                status_code=404,
                detail=f"指定された音声ライブラリ {library_id} はインストールされていません。",
            )

        # アンインストール許可フラグのバリデーション
        if not installed_libraries[library_id].uninstallable:
            raise HTTPException(
                status_code=403,
                detail=f"指定された音声ライブラリ {library_id} はアンインストールできません。",
            )

        # ディレクトリ削除によるアンインストール
        try:
            shutil.rmtree(self.library_root_dir / library_id)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"指定された音声ライブラリ {library_id} の削除に失敗しました。",
            )
