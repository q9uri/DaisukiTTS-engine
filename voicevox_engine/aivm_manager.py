import json
import shutil
import zipfile
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException
from pydantic import ValidationError
from semver.version import Version

from voicevox_engine.model import AivmInfo, AivmManifest

__all__ = ["AivmManager"]

INFO_FILE = "metas.json"


class AivmManager:
    """
    AIVM (Aivis Voice Model) 音声合成モデルを管理するクラス
    VOICEVOX ENGINE の LibraryManager がベースだが、AivisSpeech Engine 向けに大幅に改変されている
    """

    def __init__(
        self,
        installed_aivm_dir: Path,
        supported_aivm_version: str | None,
        engine_name: str,
        engine_uuid: str,
    ):
        self.installed_aivm_dir = installed_aivm_dir
        self.installed_aivm_dir.mkdir(exist_ok=True)
        if supported_aivm_version is not None:
            self.supported_aivm_version = Version.parse(supported_aivm_version)
        else:
            # supported_aivm_version が None の時は 0.0.0 として扱う
            self.supported_aivm_version = Version.parse("0.0.0")
        self.engine_name = engine_name
        self.engine_uuid = engine_uuid

    def get_installed_aivm_infos(self) -> dict[str, AivmInfo]:
        """
        インストール済み音声合成モデルの情報を取得する

        Returns
        -------
        aivm_infos : Dict[str, AivmInfo]
            インストール済み音声合成モデルの情報
        """

        aivm_infos: dict[str, AivmInfo] = {}
        for aivm_dir in self.installed_aivm_dir.iterdir():
            if aivm_dir.is_dir():
                aivm_uuid = aivm_dir.name
                with open(aivm_dir / INFO_FILE, encoding="utf-8") as f:
                    info = json.load(f)
                aivm_infos[aivm_uuid] = AivmInfo(**info)

        return aivm_infos

    def install_aivm(self, aivm_uuid: str, file: BinaryIO) -> Path:
        """
        音声合成モデルパッケージファイル (`.aivm`) をインストールする

        Parameters
        ----------
        aivm_uuid : str
            AIVM ファイルに紐づくモデル UUID (aivm_manifest.json に記載されているものと同一)
        file : BytesIO
            AIVM ファイルのバイナリ

        Returns
        -------
        model_dir : Path
            インストール済みライブラリのディレクトリパス
        """

        # ライブラリディレクトリの生成
        library_dir = self.installed_aivm_dir / aivm_uuid
        library_dir.mkdir(exist_ok=True)

        # zipファイル形式のバリデーション
        if not zipfile.is_zipfile(file):
            raise HTTPException(
                status_code=422,
                detail=f"音声合成モデル {aivm_uuid} は不正なファイルです。",
            )

        with zipfile.ZipFile(file) as zf:
            if zf.testzip() is not None:
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデル {aivm_uuid} は不正なファイルです。",
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
                    detail=f"指定された音声合成モデル {aivm_uuid} に aivm_manifest.json が存在しません。",
                )
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} の aivm_manifest.json は不正です。",
                )

            # マニフェスト形式のバリデーション
            try:
                aivm_manifest = AivmManifest.model_validate(raw_aivm_manifest)
            except ValidationError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} の aivm_manifest.json に不正なデータが含まれています。",
                )

            # ライブラリバージョンのバリデーション
            if not Version.is_valid(aivm_manifest.version):
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} の version が不正です。",
                )

            # マニフェストバージョンのバリデーション
            try:
                aivm_manifest_version = Version.parse(aivm_manifest.manifest_version)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} の manifest_version が不正です。",
                )
            if aivm_manifest_version > self.supported_aivm_version:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} は未対応です。",
                )

            # ライブラリ-エンジン対応のバリデーション
            if aivm_manifest.engine_uuid != self.engine_uuid:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} は {self.engine_name} 向けではありません。",
                )

            # モデル UUID が一致するかのバリデーション
            if aivm_manifest.uuid != aivm_uuid:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデルの UUID {aivm_uuid} が aivm_manifest.json の記述と一致しません。",
                )

            # 展開によるインストール
            zf.extractall(library_dir)

        return library_dir

    def uninstall_aivm(self, aivm_uuid: str) -> None:
        """
        インストール済み AIVM ライブラリをアンインストールする

        Parameters
        ----------
        aivm_uuid : str
            AIVM ファイルに紐づくモデル UUID (aivm_manifest.json に記載されているものと同一)
        """

        # 対象ライブラリがインストール済みであることの確認
        installed_aivm_infos = self.get_installed_aivm_infos()
        if aivm_uuid not in installed_aivm_infos.keys():
            raise HTTPException(
                status_code=404,
                detail=f"指定された音声合成モデル {aivm_uuid} はインストールされていません。",
            )

        # ディレクトリ削除によるアンインストール
        try:
            shutil.rmtree(self.installed_aivm_dir / aivm_uuid)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"指定された音声合成モデル {aivm_uuid} の削除に失敗しました。",
            )
