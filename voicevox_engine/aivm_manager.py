# flake8: noqa

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


class AivmManager:
    """
    AIVM (Aivis Voice Model) 音声合成モデルを管理するクラス
    VOICEVOX ENGINE の LibraryManager がベースだが、AivisSpeech Engine 向けに大幅に改変されている
    -----
    AIVM ファイルは音声合成モデルのパッケージファイルであり、以下の構成を持つ
    VOICEVOX の UI には立ち絵が表示されるが、立ち絵は見栄えが悪い上にユーザー側での用意が難しいため、AivisSpeech ではアイコンのみの表示に変更されている
    - aivm_manifest.json : 音声合成モデルのメタデータを記述した JSON マニフェストファイル
    - config.json : Style-Bert-VITS2 のハイパーパラメータを記述した JSON ファイル
    - model.safetensors : Style-Bert-VITS2 のモデルファイル
    - style_vectors.npy : Style-Bert-VITS2 のスタイルベクトルファイル
    - assets/
        - (speaker_uuid: aivm_manifest.json に記載の UUID)/: 話者ごとのアセット
            - icon.png : 話者 (デフォルトスタイル) のアイコン画像 (正方形)
            - voice_sample_(01~99).wav : 話者 (デフォルトスタイル) の音声サンプル
            - terms.md : 話者の利用規約
            - style-(style_id: aivm_manifest.json に記載の 0 から始まる連番 ID)/: スタイルごとのアセット (省略時はデフォルトスタイルのものが使われる)
                - icon.png : デフォルト以外の各スタイルごとのアイコン画像 (正方形)
                - voice_sample_(01~99).wav : デフォルト以外の各スタイルごとの音声サンプル
    """

    MANIFEST_FILE: str = "aivm_manifest.json"
    SUPPORTED_MANIFEST_VERSION: Version = Version.parse("1.0.0")
    SUPPORTED_ARCHITECTURES: list[str] = ["Style-Bert-VITS2"]

    def __init__(self, installed_aivm_dir: Path):
        self.installed_aivm_dir = installed_aivm_dir
        self.installed_aivm_dir.mkdir(exist_ok=True)

    def get_installed_aivm_infos(self) -> dict[str, AivmInfo]:
        """
        インストール済み音声合成モデルの情報を取得する

        Returns
        -------
        aivm_infos : dict[str, AivmInfo]
            インストール済み音声合成モデルの情報
        """

        aivm_infos: dict[str, AivmInfo] = {}
        # for aivm_dir in self.installed_aivm_dir.iterdir():
        #     if aivm_dir.is_dir():
        #         aivm_uuid = aivm_dir.name
        #         with open(aivm_dir / INFO_FILE, encoding="utf-8") as f:
        #             info = json.load(f)
        #         aivm_infos[aivm_uuid] = AivmInfo(**info)

        return aivm_infos

    def install_aivm(self, aivm_uuid: str, file: BinaryIO) -> Path:
        """
        音声合成モデルパッケージファイル (`.aivm`) をインストールする

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルのUUID (aivm_manifest.json に記載されているものと同一)
        file : BytesIO
            AIVM ファイルのバイナリ

        Returns
        -------
        model_dir : Path
            インストール済みライブラリのディレクトリパス
        """

        # インストール先ディレクトリの生成
        install_dir = self.installed_aivm_dir / aivm_uuid
        install_dir.mkdir(exist_ok=True)

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
                    zf.read(self.MANIFEST_FILE).decode("utf-8")
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

            # マニフェストバージョンのバリデーション
            try:
                aivm_manifest_version = Version.parse(aivm_manifest.manifest_version)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} の manifest_version ({aivm_manifest.manifest_version}) は不正です。",
                )
            if aivm_manifest_version > self.SUPPORTED_MANIFEST_VERSION:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} は未対応です。",
                )

            # 音声合成モデルのバージョンのバリデーション
            if not Version.is_valid(aivm_manifest.version):
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} の version ({aivm_manifest.version}) は不正です。",
                )

            # 音声合成モデルのアーキテクチャのバリデーション
            if aivm_manifest.architecture not in self.SUPPORTED_ARCHITECTURES:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデル {aivm_uuid} の architecture ({aivm_manifest.architecture}) は未対応です。",
                )

            # 音声合成モデルの UUID のバリデーション
            if aivm_manifest.uuid != aivm_uuid:
                raise HTTPException(
                    status_code=422,
                    detail=f"指定された音声合成モデルの UUID {aivm_uuid} が aivm_manifest.json の uuid ({aivm_manifest.uuid}) と一致しません。",
                )

            # 展開によるインストール
            zf.extractall(install_dir)

        return install_dir

    def uninstall_aivm(self, aivm_uuid: str) -> None:
        """
        インストール済み音声合成モデルをアンインストールする

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)
        """

        # 対象の音声合成モデルがインストール済みであることの確認
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
