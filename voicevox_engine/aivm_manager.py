# flake8: noqa

import base64
import json
import shutil
import zipfile
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException
from pydantic import ValidationError
from semver.version import Version

from .logging import logger
from .metas.Metas import (
    Speaker,
    SpeakerInfo,
    SpeakerStyle,
    SpeakerSupportedFeatures,
    SpeakerSupportPermittedSynthesisMorphing,
    StyleId,
    StyleInfo,
)
from .model import (
    AivmInfo,
    AivmInfoSpeaker,
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
)

__all__ = ["AivmManager"]


class AivmManager:
    """
    AIVM (Aivis Voice Model) 音声合成モデルを管理するクラス
    VOICEVOX ENGINE の LibraryManager がベースだが、AivisSpeech Engine 向けに大幅に改変されている
    -----
    AIVM ファイルは音声合成モデルのパッケージファイルであり、以下の構成を持つ
    VOICEVOX 本家では話者の立ち絵が表示されるが、立ち絵は見栄えが悪い上にユーザー側での用意が難しいため、AivisSpeech ではアイコンのみの表示に変更されている
    - aivm_manifest.json : 音声合成モデルのメタデータを記述した JSON マニフェストファイル
    - config.json : Style-Bert-VITS2 のハイパーパラメータを記述した JSON ファイル
    - model.safetensors : Style-Bert-VITS2 のモデルファイル
    - style_vectors.npy : Style-Bert-VITS2 のスタイルベクトルファイル
    - speaker_(speaker_uuid: aivm_manifest.json に記載の UUID)/: 話者ごとのアセット
        - icon.png : 話者 (デフォルトスタイル) のアイコン画像 (正方形)
        - voice_sample_(01~99).wav : 話者 (デフォルトスタイル) の音声サンプル
        - terms.md : 話者の利用規約
        - style_(style_id: aivm_manifest.json に記載の 0 から始まる連番 ID)/: スタイルごとのアセット (省略時はデフォルトスタイルのものが使われる)
            - icon.png : デフォルト以外の各スタイルごとのアイコン画像 (正方形)
            - voice_sample_(01~99).wav : デフォルト以外の各スタイルごとの音声サンプル
    """

    MANIFEST_FILE: str = "aivm_manifest.json"
    SUPPORTED_MANIFEST_VERSION: Version = Version.parse("1.0.0")
    SUPPORTED_MODEL_ARCHITECTURES: list[str] = ["Style-Bert-VITS2"]

    def __init__(self, installed_aivm_dir: Path):
        self.installed_aivm_dir = installed_aivm_dir
        self.installed_aivm_dir.mkdir(exist_ok=True)

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

        # 話者名でソート
        speakers = sorted(speakers, key=lambda x: x.name)

        return speakers

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

    def get_installed_aivm_infos(self) -> dict[str, AivmInfo]:
        """
        すべてのインストール済み音声合成モデルの情報を取得する

        Returns
        -------
        aivm_infos : dict[str, AivmInfo]
            インストール済み音声合成モデルの情報 (キー: AIVM UUID, 値: AivmInfo)
        """

        aivm_infos: dict[str, AivmInfo] = {}
        for aivm_dir in self.installed_aivm_dir.iterdir():
            if aivm_dir.is_dir():
                aivm_uuid = aivm_dir.name.replace("aivm_", "")

                # AIVM マニフェストを取得し、仮で AivmInfo に変換
                ## 話者情報は後で追加するため、空リストを渡す
                try:
                    aivm_manifest = self.get_aivm_manifest(aivm_uuid)
                except HTTPException as e:
                    logger.warning(f"Failed to get AivmManifest of {aivm_uuid}: {e.detail}")  # fmt: skip
                    continue
                aivm_info = AivmInfo(
                    name=aivm_manifest.name,
                    description=aivm_manifest.description,
                    model_architecture=aivm_manifest.model_architecture,
                    uuid=aivm_manifest.uuid,
                    version=aivm_manifest.version,
                    speakers=[],
                )

                # 万が一対応していないアーキテクチャの音声合成モデルの場合は除外
                if aivm_manifest.model_architecture not in self.SUPPORTED_MODEL_ARCHITECTURES:  # fmt: skip
                    logger.warning(f"TTS model {aivm_uuid} has unsupported model architecture {aivm_manifest.model_architecture}. Skipping.")  # fmt: skip
                    continue

                # 話者情報を AivmInfoSpeaker に変換し、AivmInfo.speakers に追加
                for speaker_manifest in aivm_manifest.speakers:
                    speaker_uuid = speaker_manifest.uuid
                    speaker_dir = aivm_dir / f"speaker_{speaker_uuid}"

                    # AivisSpeech Engine は日本語のみをサポートするため、日本語をサポートしない話者は除外
                    ## supported_languages に大文字が設定されている可能性もあるため、小文字に変換して比較
                    if "ja" not in [lang.lower() for lang in speaker_manifest.supported_languages]:  # fmt: skip
                        logger.warning(f"Speaker {speaker_uuid} of TTS model {aivm_uuid} does not support Japanese. Skipping.")  # fmt: skip
                        continue

                    # 話者のディレクトリが存在しない場合
                    if not speaker_dir.exists():
                        raise HTTPException(
                            status_code=500,
                            detail=f"音声合成モデル {aivm_uuid} の話者 {speaker_uuid} のディレクトリが存在しません。",
                        )

                    # デフォルトスタイルのアセットと利用規約 (Markdown) のパスを取得
                    default_style_icon_path = speaker_dir / "icon.png"
                    if not default_style_icon_path.exists():
                        raise HTTPException(
                            status_code=500,
                            detail=f"音声合成モデル {aivm_uuid} の話者 {speaker_uuid} に icon.png が存在しません。",
                        )
                    terms_path = speaker_dir / "terms.md"
                    if not terms_path.exists():
                        raise HTTPException(
                            status_code=500,
                            detail=f"音声合成モデル {aivm_uuid} の話者 {speaker_uuid} に terms.md が存在しません。",
                        )
                    default_style_voice_sample_paths = sorted(list(speaker_dir.glob("voice_sample_*.wav")))  # fmt: skip

                    # スタイルごとにアセットを取得
                    speaker_styles: list[SpeakerStyle] = []
                    style_infos: list[StyleInfo] = []
                    for style_manifest in speaker_manifest.styles:

                        # AIVM マニフェスト内のスタイル ID を VOICEVOX ENGINE 互換の StyleId に変換
                        style_id = self.local_style_id_to_style_id(style_manifest.id, speaker_uuid)  # fmt: skip

                        # スタイルごとのディレクトリが存在する場合はアセットのパスを取得
                        style_dir = speaker_dir / f"style_{style_id}"
                        if style_dir.exists() and style_dir.is_dir():
                            style_icon_path = style_dir / "icon.png"
                            if not style_icon_path.exists():
                                raise HTTPException(
                                    status_code=500,
                                    detail=f"音声合成モデル {aivm_uuid} の話者 {speaker_uuid} のスタイル {style_id} に icon.png が存在しません。",
                                )
                            style_voice_sample_paths = sorted(list(style_dir.glob("voice_sample_*.wav")))  # fmt: skip

                        # スタイルディレクトリが存在しない場合はデフォルトスタイルのアセットのパスを使う
                        ## デフォルトスタイル (ID: 0) はスタイルごとのディレクトリが作成されないため、常にこの分岐に入る
                        else:
                            style_icon_path = default_style_icon_path
                            style_voice_sample_paths = default_style_voice_sample_paths

                        # SpeakerStyle の作成
                        speaker_style = SpeakerStyle(
                            id=style_id,
                            name=style_manifest.name,
                            # AivisSpeech は仮称音声合成に対応しないので talk で固定
                            type="talk",
                        )
                        speaker_styles.append(speaker_style)

                        # StyleInfo の作成
                        style_info = StyleInfo(
                            id=style_id,
                            # アイコン画像を Base64 エンコードして文字列化
                            icon=base64.b64encode(style_icon_path.read_bytes()).decode("utf-8"),
                            # 立ち絵を省略
                            ## VOICEVOX 本家では portrait に立ち絵が入るが、AivisSpeech では敢えてアイコン画像のみを設定する
                            portrait=None,
                            # 音声サンプルを Base64 エンコードして文字列化
                            voice_samples=[
                                base64.b64encode(sample_path.read_bytes()).decode("utf-8")
                                for sample_path in style_voice_sample_paths
                            ],
                        )  # fmt: skip
                        style_infos.append(style_info)

                    # AivmInfoSpeaker の作成
                    aivm_info_speaker = AivmInfoSpeaker(
                        speaker=Speaker(
                            speaker_uuid=speaker_uuid,
                            name=speaker_manifest.name,
                            styles=speaker_styles,
                            version=speaker_manifest.version,
                            # AivisSpeech Engine では全話者に対し常にモーフィング機能を無効化する
                            ## Style-Bert-VITS2 の仕様上音素長を一定にできず、話者ごとに発話タイミングがずれてまともに合成できないため
                            supported_features=SpeakerSupportedFeatures(
                                permitted_synthesis_morphing=SpeakerSupportPermittedSynthesisMorphing.NOTHING,
                            ),
                        ),
                        speaker_info=SpeakerInfo(
                            # 利用規約をそのまま読み取って文字列に格納
                            policy=terms_path.read_text(encoding="utf-8"),
                            # アイコン画像を Base64 エンコードして文字列化
                            ## VOICEVOX 本家では portrait に立ち絵が入るが、AivisSpeech では敢えてアイコン画像を設定する
                            portrait=base64.b64encode(default_style_icon_path.read_bytes()).decode("utf-8"),
                            style_infos=style_infos,
                        ),
                    )  # fmt: skip
                    aivm_info.speakers.append(aivm_info_speaker)

                # 完成した AivmInfo を UUID をキーとして追加
                aivm_infos[aivm_uuid] = aivm_info

        # 音声合成モデル名でソート
        aivm_infos = dict(sorted(aivm_infos.items(), key=lambda x: x[1].name))

        return aivm_infos

    def get_aivm_manifest(self, aivm_uuid: str) -> AivmManifest:
        """
        AIVM (Aivis Voice Model) マニフェストを取得する

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        aivm_manifest : AivmManifest
            AIVM (Aivis Voice Model) マニフェスト
        """

        # 対象の音声合成モデルがインストール済みであることの確認
        aivm_dir = self.installed_aivm_dir / f"aivm_{aivm_uuid}"
        if aivm_dir.is_dir() is False:
            raise HTTPException(
                status_code=404,
                detail=f"音声合成モデル {aivm_uuid} はインストールされていません。",
            )

        # マニフェストファイルの存在確認
        aivm_manifest_path = aivm_dir / self.MANIFEST_FILE
        if aivm_manifest_path.is_file() is False:
            raise HTTPException(
                status_code=500,
                detail=f"音声合成モデル {aivm_uuid} に aivm_manifest.json が存在しません。",
            )

        # マニフェストファイルの読み込み
        try:
            with open(aivm_manifest_path, mode="r", encoding="utf-8") as f:
                raw_aivm_manifest = json.load(f)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"音声合成モデル {aivm_uuid} の aivm_manifest.json の読み込みに失敗しました。",
            )

        # マニフェスト形式のバリデーション
        try:
            aivm_manifest = AivmManifest.model_validate(raw_aivm_manifest)
        except ValidationError:
            raise HTTPException(
                status_code=500,
                detail=f"音声合成モデル {aivm_uuid} の aivm_manifest.json に不正なデータが含まれています。",
            )

        return aivm_manifest

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
            AIVM (Aivis Voice Model) マニフェスト
        aivm_manifest_speaker : AivmManifestSpeaker
            AIVM (Aivis Voice Model) マニフェスト内の話者
        aivm_manifest_style : AivmManifestSpeakerStyle
            AIVM (Aivis Voice Model) マニフェスト内のスタイル
        """

        # fmt: off
        aivm_infos = self.get_installed_aivm_infos()
        for aivm_info in aivm_infos.values():
            for aivm_info_speaker in aivm_info.speakers:
                for aivm_info_speaker_style in aivm_info_speaker.speaker.styles:
                    if aivm_info_speaker_style.id == style_id:
                        # ここでスタイル ID が示す音声合成モデルに対応する AivmManifest を特定
                        aivm_manifest = self.get_aivm_manifest(aivm_info.uuid)
                        for aivm_manifest_speaker in aivm_manifest.speakers:
                            # ここでスタイル ID が示す話者に対応する AivmManifestSpeaker を特定
                            if aivm_manifest_speaker.uuid == aivm_info_speaker.speaker.speaker_uuid:
                                for aivm_manifest_style in aivm_manifest_speaker.styles:
                                    # ここでスタイル ID が示すスタイルに対応する AivmManifestSpeakerStyle を特定
                                    local_style_id = self.style_id_to_local_style_id(style_id, aivm_manifest_speaker.uuid)
                                    if aivm_manifest_style.id == local_style_id:
                                        # すべて取得できたので値を返す
                                        return aivm_manifest, aivm_manifest_speaker, aivm_manifest_style

        raise HTTPException(
            status_code=404,
            detail=f"スタイル {style_id} は存在しません。",
        )

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
        install_dir = self.installed_aivm_dir / f"aivm_{aivm_uuid}"
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
                    detail=f"音声合成モデル {aivm_uuid} に aivm_manifest.json が存在しません。",
                )
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデル {aivm_uuid} の aivm_manifest.json は不正です。",
                )

            # マニフェスト形式のバリデーション
            try:
                aivm_manifest = AivmManifest.model_validate(raw_aivm_manifest)
            except ValidationError:
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデル {aivm_uuid} の aivm_manifest.json に不正なデータが含まれています。",
                )

            # マニフェストバージョンのバリデーション
            try:
                aivm_manifest_version = Version.parse(aivm_manifest.manifest_version)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデル {aivm_uuid} の manifest_version ({aivm_manifest.manifest_version}) は不正です。",
                )
            if aivm_manifest_version > self.SUPPORTED_MANIFEST_VERSION:
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデル {aivm_uuid} は未対応です。",
                )

            # 音声合成モデルのバージョンのバリデーション
            if not Version.is_valid(aivm_manifest.version):
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデル {aivm_uuid} の version ({aivm_manifest.version}) は不正です。",
                )

            # 音声合成モデルのアーキテクチャのバリデーション
            if aivm_manifest.model_architecture not in self.SUPPORTED_MODEL_ARCHITECTURES:  # fmt: skip
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデル {aivm_uuid} の architecture ({aivm_manifest.model_architecture}) は未対応です。",
                )

            # 音声合成モデルの UUID のバリデーション
            if aivm_manifest.uuid != aivm_uuid:
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデルの UUID {aivm_uuid} が aivm_manifest.json の uuid ({aivm_manifest.uuid}) と一致しません。",
                )

            # AIVM ファイル内に必須のファイルが存在するか確認
            REQUIRED_FILES: list[str] = [
                "config.json",
                "model.safetensors",
                "style_vectors.npy",
            ]
            for aivm_manifest_speaker in aivm_manifest.speakers:
                # 話者のデフォルトスタイルのアイコンと利用規約は必須
                # デフォルトスタイル以外のスタイルのアイコンと、ボイスサンプルは任意
                speaker_dir = f"speaker_{aivm_manifest_speaker.uuid}/"
                REQUIRED_FILES.append(speaker_dir + "icon.png")
                REQUIRED_FILES.append(speaker_dir + "terms.md")
            if not all([file in zf.namelist() for file in REQUIRED_FILES]):
                raise HTTPException(
                    status_code=422,
                    detail=f"音声合成モデル {aivm_uuid} に必要なファイルが存在しません。",
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
                detail=f"音声合成モデル {aivm_uuid} はインストールされていません。",
            )

        # ディレクトリ削除によるアンインストール
        aivm_dir = self.installed_aivm_dir / f"aivm_{aivm_uuid}"
        try:
            shutil.rmtree(aivm_dir)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"音声合成モデル {aivm_uuid} の削除に失敗しました。",
            )

    @staticmethod
    def local_style_id_to_style_id(local_style_id: int, speaker_uuid: str) -> StyleId:
        """
        AIVM マニフェスト内のローカルなスタイル ID を VOICEVOX ENGINE 互換のグローバルに一意な StyleId に変換する

        Parameters
        ----------
        manifest_style_id : int
            AIVM マニフェスト内のローカルなスタイル ID
        speaker_uuid : str
            話者の UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        style_id : StyleId
            VOICEVOX ENGINE 互換のグローバルに一意なスタイル ID
        """

        # AIVM マニフェスト内のスタイル ID は、話者ごとにローカルな 0 から始まる連番になっている
        # この値は config.json に記述されているハイパーパラメータの data.style2id の値と一致する
        # 一方 VOICEVOX ENGINE は互換性問題？による歴史的事情でスタイル ID を音声合成 API に渡す形となっており、
        # スタイル ID がグローバルに一意になっていなければならない
        # そこで、数値化した話者 UUID にスタイル ID を組み合わせることで、一意なスタイル ID を生成する
        # 2**53 - 1 は JavaScript の Number.MAX_SAFE_INTEGER に合わせた値
        uuid_int = int(speaker_uuid.replace("-", ""), 16)
        return StyleId((uuid_int % (2**53 - 1)) + local_style_id)

    @staticmethod
    def style_id_to_local_style_id(style_id: StyleId, speaker_uuid: str) -> int:
        """
        VOICEVOX ENGINE 互換のグローバルに一意な StyleId を AIVM マニフェスト内のローカルなスタイル ID に変換する

        Parameters
        ----------
        style_id : StyleId
            VOICEVOX ENGINE 互換のグローバルに一意なスタイル ID
        speaker_uuid : str
            話者の UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        local_style_id : int
            AIVM マニフェスト内のローカルなスタイル ID
        """

        # local_style_id_to_style_id() の逆変換
        uuid_int = int(speaker_uuid.replace("-", ""), 16)
        return (style_id - (uuid_int % (2**53 - 1))) % (2**53 - 1)
