import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException
from pydantic import ValidationError
from semver.version import Version

from voicevox_engine.model import AivmManifest, InstalledLibraryInfo

__all__ = ["LibraryManager"]

INFO_FILE = "metas.json"


class LibraryManager:
    """
    AIVM (Aivis Voice Model) 音声合成モデルの管理
    「ライブラリ」という名称のままなのは VOICEVOX ENGINE からの大幅な改変を避けるため
    """

    def __init__(
        self,
        installed_models_dir: Path,
        supported_aivm_version: str | None,
        brand_name: str,
        engine_name: str,
        engine_uuid: str,
    ):
        self.installed_models_dir = installed_models_dir
        self.installed_models_dir.mkdir(exist_ok=True)
        if supported_aivm_version is not None:
            self.supported_aivm_version = Version.parse(supported_aivm_version)
        else:
            # supported_aivm_version が None の時は 0.0.0 として扱う
            self.supported_aivm_version = Version.parse("0.0.0")
        self.engine_brand_name = brand_name
        self.engine_name = engine_name
        self.engine_uuid = engine_uuid

    def installed_models(self) -> dict[str, InstalledLibraryInfo]:
        """
        インストール済み音声合成モデルの情報を取得する

        Returns
        -------
        library : Dict[str, InstalledLibraryInfo]
            インストール済み音声合成モデルの情報
        """

        library: dict[str, InstalledLibraryInfo] = {}
        for library_dir in self.installed_models_dir.iterdir():
            if library_dir.is_dir():
                # 音声合成モデル情報の取得 from `library_root_dir / f"{library_uuid}" / "metas.json"`
                library_uuid = os.path.basename(library_dir)
                with open(library_dir / INFO_FILE, encoding="utf-8") as f:
                    info = json.load(f)
                library[library_uuid] = InstalledLibraryInfo(**info, uninstallable=True)

        return library

    def install_models(self, model_uuid: str, file: BinaryIO) -> Path:
        """
        音声合成モデルパッケージファイル (`.aivm`) をインストールする

        Parameters
        ----------
        model_uuid : str
            AIVM ファイルに紐づくモデル UUID (aivm_manifest.json に記載されているものと同一)
        file : BytesIO
            AIVM ファイルのバイナリ

        Returns
        -------
        model_dir : Path
            インストール済みライブラリのディレクトリパス
        """

        # ライブラリディレクトリの生成
        library_dir = self.installed_models_dir / model_uuid
        library_dir.mkdir(exist_ok=True)

        # zipファイル形式のバリデーション
        if not zipfile.is_zipfile(file):
            raise HTTPException(
                status_code=422,
                detail=f"音声ライブラリ {model_uuid} は不正なファイルです。",
            )

        with zipfile.ZipFile(file) as zf:
            if zf.testzip() is not None:
                raise HTTPException(
                    status_code=422,
                    detail=f"音声ライブラリ {model_uuid} は不正なファイルです。",
                )

            # マニフェストファイルの存在とファイル形式をバリデーション
            raw_aivm_manifest = None
            try:
                raw_aivm_manifest = json.loads(
                    zf.read("aivm_manifest.json").decode("utf-8")
                )
            except KeyError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {model_uuid} に aivm_manifest.json が存在しません。",
                )
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {model_uuid} の aivm_manifest.json は不正です。",
                )

            # マニフェスト形式のバリデーション
            try:
                aivm_manifest = AivmManifest.model_validate(raw_aivm_manifest)
            except ValidationError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {model_uuid} の aivm_manifest.json に不正なデータが含まれています。",
                )

            # ライブラリバージョンのバリデーション
            if not Version.is_valid(aivm_manifest.version):
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {model_uuid} の version が不正です。",
                )

            # マニフェストバージョンのバリデーション
            try:
                aivm_manifest_version = Version.parse(aivm_manifest.manifest_version)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {model_uuid} の manifest_version が不正です。",
                )
            if aivm_manifest_version > self.supported_aivm_version:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {model_uuid} は未対応です。",
                )

            # ライブラリ-エンジン対応のバリデーション
            if aivm_manifest.engine_uuid != self.engine_uuid:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリ {model_uuid} は {self.engine_name} 向けではありません。",
                )

            # モデル UUID が一致するかのバリデーション
            if aivm_manifest.uuid != model_uuid:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声ライブラリの UUID {model_uuid} が aivm_manifest.json の記述と一致しません。",
                )

            # 展開によるインストール
            zf.extractall(library_dir)

        return library_dir

    def uninstall_models(self, library_id: str) -> None:
        """
        インストール済み AIVM ライブラリをアンインストールする

        Parameters
        ----------
        library_id : str
            インストール対象のライブラリ ID
        """

        # 対象ライブラリがインストール済みであることの確認
        installed_libraries = self.installed_models()
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
            shutil.rmtree(self.installed_models_dir / library_id)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"指定された音声ライブラリ {library_id} の削除に失敗しました。",
            )
