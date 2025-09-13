"""AIVM (Aivis Voice Model) 仕様に準拠した音声合成モデルと AIVM マニフェストを管理するクラス"""

import re
import time
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Final

import aivmlib
import httpx
from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
)
from fastapi import HTTPException

from voicevox_engine.aivm_infos_repository import AivmInfosRepository
from voicevox_engine.logging import logger
from voicevox_engine.metas.Metas import Speaker, SpeakerInfo, StyleId
from voicevox_engine.metas.MetasStore import Character
from voicevox_engine.model import AivmInfo
from voicevox_engine.utility.user_agent_utility import generate_user_agent

__all__ = ["AivmManager"]


class AivmManager:
    """
    AIVM (Aivis Voice Model) 仕様に準拠した音声合成モデルと AIVM マニフェストを管理するクラス。

    VOICEVOX ENGINE における MetasStore の役割を代替する。(AivisSpeech Engine では MetasStore は無効化されている)
    AivisSpeech はインストールサイズを削減するため、AIVMX ファイルにのみ対応している。
    ref: https://github.com/Aivis-Project/aivmlib#aivm-specification
    """

    # デフォルトでインストールされる音声合成モデルの UUID
    DEFAULT_MODEL_UUIDS: Final[list[str]] = [
        "a59cb814-0083-4369-8542-f51a29e72af7",
    ]

    def __init__(self, installed_models_dir: Path):
        """
        AivmManager のコンストラクタ

        Parameters
        ----------
        installed_models_dir : Path
            AIVMX ファイルのインストール先ディレクトリ
        """

        # インストール先ディレクトリが存在しなければここで作成
        self.installed_models_dir = installed_models_dir
        self.installed_models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Models directory: {self.installed_models_dir}")

        # リポジトリを初期化
        ## この時点で前回起動時に作成したキャッシュがあればそこから即座に情報を読み込む
        ## キャッシュがない場合はコンストラクタで同期的にスキャンを行い、スキャン完了次第リポジトリの初期化が完了する
        self._repository = AivmInfosRepository(self.installed_models_dir)

        # まだ一つも音声合成モデルがインストールされていない場合、デフォルトモデルをインストール
        # メタデータの読み取りに失敗したなどで情報を取得できなかったモデルはインストールされていないとみなす
        current_installed_aivm_infos = self._repository.get_installed_aivm_infos()
        if len(current_installed_aivm_infos) == 0:
            logger.warning("No models are installed.")
        else:
            logger.info("Installed models:")
            for aivm_info in current_installed_aivm_infos.values():
                logger.info(f"- {aivm_info.manifest.name} ({aivm_info.manifest.uuid})")

    def get_characters(self) -> list[Character]:
        """
        すべてのインストール済み音声合成モデル内の話者の一覧を Character 型で取得する (MetasStore 互換用)

        Returns
        -------
        characters : list[Character]
            インストール済み音声合成モデル内の話者の一覧
        """

        speakers = self.get_speakers()
        characters: list[Character] = []
        for speaker in speakers:
            character = Character(
                name=speaker.name,
                uuid=speaker.speaker_uuid,
                # AivisSpeech Engine では talk スタイルのみがサポートされる
                talk_styles=speaker.styles,
                # AivisSpeech Engine では歌唱音声合成はサポートされていない
                sing_styles=[],
                version=speaker.version,
                supported_features=speaker.supported_features,
            )
            characters.append(character)

        # 既に get_speakers() で話者名でソートされているのでそのまま返す
        return characters

    def get_speakers(self) -> list[Speaker]:
        """
        すべてのインストール済み音声合成モデル内の話者の一覧を取得する

        Returns
        -------
        speakers : list[Speaker]
            インストール済み音声合成モデル内の話者の一覧
        """

        aivm_infos = self.get_installed_aivm_infos()
        speakers: list[Speaker] = []
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                speakers.append(aivm_info_speaker.speaker)

        # 話者名でソートしてから返す
        return sorted(speakers, key=lambda x: x.name)

    def get_speaker_info(self, speaker_uuid: str) -> SpeakerInfo:
        """
        インストール済み音声合成モデル内の話者の追加情報を取得する

        Parameters
        ----------
        speaker_uuid : str
            話者の UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        speaker_info : SpeakerInfo
            話者の追加情報
        """

        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                if aivm_info_speaker.speaker.speaker_uuid == speaker_uuid:
                    return aivm_info_speaker.speaker_info

        raise HTTPException(
            status_code=404,
            detail=f"話者 {speaker_uuid} はインストールされていません。",
        )

    def get_aivm_info(self, aivm_uuid: str) -> AivmInfo:
        """
        音声合成モデルの UUID から AIVMX ファイルの情報を取得する

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        aivm_info : AivmInfo
            AIVMX ファイルの情報
        """

        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            if str(aivm_info.manifest.uuid) == aivm_uuid:
                return aivm_info

        raise HTTPException(
            status_code=404,
            detail=f"音声合成モデル {aivm_uuid} はインストールされていません。",
        )

    def get_aivm_manifest_from_style_id(
        self, style_id: StyleId
    ) -> tuple[AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle]:
        """
        スタイル ID に対応する AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle を取得する

        Parameters
        ----------
        style_id : StyleId
            スタイル ID

        Returns
        -------
        aivm_manifest : AivmManifest
            AIVM マニフェスト
        aivm_manifest_speaker : AivmManifestSpeaker
            AIVM マニフェスト内の話者
        aivm_manifest_style : AivmManifestSpeakerStyle
            AIVM マニフェスト内のスタイル
        """

        # fmt: off
        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                for aivm_info_speaker_style in aivm_info_speaker.speaker.styles:
                    if aivm_info_speaker_style.id == style_id:
                        # ここでスタイル ID が示す音声合成モデルに対応する AivmManifest を特定
                        aivm_manifest = aivm_info.manifest
                        for aivm_manifest_speaker in aivm_manifest.speakers:
                            # ここでスタイル ID が示す話者に対応する AivmManifestSpeaker を特定
                            if str(aivm_manifest_speaker.uuid) == aivm_info_speaker.speaker.speaker_uuid:
                                for aivm_manifest_style in aivm_manifest_speaker.styles:
                                    # ここでスタイル ID が示すスタイルに対応する AivmManifestSpeakerStyle を特定
                                    local_style_id = self._repository.style_id_to_local_style_id(style_id)
                                    if aivm_manifest_style.local_id == local_style_id:
                                        # すべて取得できたので値を返す
                                        return aivm_manifest, aivm_manifest_speaker, aivm_manifest_style

        raise HTTPException(
            status_code=404,
            detail=f"スタイル {style_id} は存在しません。",
        )

    def get_installed_aivm_infos(self) -> dict[str, AivmInfo]:
        """
        すべてのインストール済み音声合成モデルの情報を取得する

        Returns
        -------
        aivm_infos : dict[str, AivmInfo]
            インストール済み音声合成モデルの情報 (キー: 音声合成モデルの UUID, 値: AivmInfo)
        """

        # リポジトリの現在の状態を返す
        return self._repository.get_installed_aivm_infos()

    def update_model_load_state(self, aivm_uuid: str, is_loaded: bool) -> None:
        """
        音声合成モデルのロード状態を更新する
        このメソッドは StyleBertVITS2TTSEngine 上でロード/アンロードが行われた際に呼び出される

        Parameters
        ----------
        aivm_uuid : str
            AIVM マニフェスト記載の UUID
        is_loaded : bool
            モデルがロードされているかどうか
        """

        # リポジトリの現在のモデルロード状態を更新する
        self._repository.update_model_load_state(aivm_uuid, is_loaded)

    def install_model(self, file: BinaryIO) -> None:
        """
        AIVMX (Aivis Voice Model for ONNX) ファイル (`.aivmx`) をインストールする

        Parameters
        ----------
        file : BinaryIO
            AIVMX ファイルのバイナリ
        """

        # AIVMX ファイルからから AIVM メタデータを取得
        try:
            aivm_metadata = aivmlib.read_aivmx_metadata(file)
            aivm_manifest = aivm_metadata.manifest
        except aivmlib.AivmValidationError as ex:
            logger.error("AIVMX file is invalid:", exc_info=ex)
            raise HTTPException(
                status_code=422,
                detail=f"指定された AIVMX ファイルの形式が正しくありません。({ex})",
            ) from ex

        # すでに同一 UUID のファイルがインストール済みの場合、同じファイルを更新する
        ## 手動で .aivmx ファイルをインストール先ディレクトリにコピーしていた (ファイル名が UUID と一致しない) 場合も更新できるよう、
        ## この場合のみ特別に更新先ファイル名を現在保存されているファイル名に変更する
        aivm_file_path = self.installed_models_dir / f"{aivm_manifest.uuid}.aivmx"
        aivm_infos = self.get_installed_aivm_infos()
        if str(aivm_manifest.uuid) in aivm_infos:
            logger.info(
                f"AIVM model {aivm_manifest.uuid} is already installed. Updating..."
            )
            # aivm_file_path を現在保存されているファイル名に変更
            aivm_file_path = aivm_infos[str(aivm_manifest.uuid)].file_path

        # マニフェストバージョンのバリデーション
        if (
            aivm_manifest.manifest_version
            not in self._repository.SUPPORTED_MANIFEST_VERSIONS
        ):
            logger.error(
                f"AIVM manifest version {aivm_manifest.manifest_version} is not supported."
            )
            raise HTTPException(
                status_code=422,
                detail=f"AIVM マニフェストバージョン {aivm_manifest.manifest_version} には対応していません。",
            )

        # 音声合成モデルのアーキテクチャのバリデーション
        if (
            aivm_manifest.model_architecture
            not in self._repository.SUPPORTED_MODEL_ARCHITECTURES
        ):
            logger.error(
                f"AIVM model architecture {aivm_manifest.model_architecture} is not supported."
            )
            raise HTTPException(
                status_code=422,
                detail=f'モデルアーキテクチャ "{aivm_manifest.model_architecture}" には対応していません。',
            )

        # BinaryIO のシークをリセット
        # ここでリセットしないとファイルの内容を読み込めない
        file.seek(0)

        # AIVMX ファイルをインストール
        ## 通常は重複防止のため "(音声合成モデルの UUID).aivmx" のフォーマットのファイル名でインストールされるが、
        ## 手動で .aivmx ファイルをインストール先ディレクトリにコピーしても一通り動作するように考慮している
        logger.info(f"Installing AIVMX file to {aivm_file_path}...")
        try:
            with open(aivm_file_path, mode="wb") as f:
                f.write(file.read())
            logger.info(f"Installed AIVMX file to {aivm_file_path}.")
        except OSError as ex:
            logger.error(
                f"Failed to write AIVMX file to {aivm_file_path}:", exc_info=ex
            )
            error_message = str(ex).lower()
            if "no space" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。ストレージ容量が不足しています。({ex})"
            elif "permission denied" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。インストール先フォルダへのアクセス権限が不足しています。({ex})"
            elif "read-only" in error_message:
                detail = f"AIVMX ファイルの書き込みに失敗しました。インストール先フォルダが読み取り専用権限になっています。({ex})"
            else:
                detail = f"AIVMX ファイルの書き込みに失敗しました。({ex})"
            raise HTTPException(
                status_code=500,
                detail=detail,
            ) from ex

        # すべてのインストール済み音声合成モデルの情報を再取得
        ## このメソッドは情報更新後、AivisHub からアップデート情報を再取得してから戻る
        self._repository.update_repository()

    def install_model_from_url(self, url: str) -> None:
        """
        指定された URL から AIVMX (Aivis Voice Model for ONNX) ファイル (`.aivmx`) をダウンロードしてインストールする

        Parameters
        ----------
        url : str
            AIVMX ファイルの URL
        """

        print("not supported")

    def update_model(self, aivm_uuid: str) -> None:
        """
        AivisHub から指定された音声合成モデルの一番新しいバージョンをダウンロードし、
        インストール済みの音声合成モデルへ上書き更新する

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)
        """

        print("not supported")

    def uninstall_model(self, aivm_uuid: str) -> None:
        """
        インストール済み音声合成モデルをアンインストールする

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)
        """

        # 対象の音声合成モデルがインストール済みかを確認
        installed_aivm_infos = self.get_installed_aivm_infos()
        if aivm_uuid not in installed_aivm_infos.keys():
            raise HTTPException(
                status_code=404,
                detail=f"音声合成モデル {aivm_uuid} はインストールされていません。",
            )

        # インストール済みの音声合成モデルの数を確認
        if len(installed_aivm_infos) <= 1:
            logger.error("AivisSpeech Engine must have at least one installed model.")
            raise HTTPException(
                status_code=400,
                detail="AivisSpeech Engine には必ず 1 つ以上の音声合成モデルがインストールされている必要があります。",
            )

        # AIVMX ファイルをアンインストール
        ## AIVMX ファイルのファイル名は必ずしも "(音声合成モデルの UUID).aivmx" になるとは限らないため、
        ## AivmInfo 内に格納されているファイルパスを使って削除する
        ## 万が一 AIVMX ファイルが存在しない場合は無視する
        logger.info(
            f"Uninstalling AIVMX file from {installed_aivm_infos[aivm_uuid].file_path}..."
        )
        installed_aivm_infos[aivm_uuid].file_path.unlink(missing_ok=True)
        logger.info(
            f"Uninstalled AIVMX file from {installed_aivm_infos[aivm_uuid].file_path}."
        )

        # すべてのインストール済み音声合成モデルの情報を再取得
        ## このメソッドは情報更新後、AivisHub からアップデート情報を再取得してから戻る
        self._repository.update_repository()
