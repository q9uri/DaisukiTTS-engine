# flake8: noqa

import glob
import hashlib
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import aivmlib
import httpx
from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    ModelArchitecture,
)
from fastapi import HTTPException

from voicevox_engine import __version__
from voicevox_engine.logging import logger
from voicevox_engine.metas.Metas import (
    Speaker,
    SpeakerInfo,
    SpeakerStyle,
    SpeakerSupportedFeatures,
    SpeakerSupportPermittedSynthesisMorphing,
    StyleId,
    StyleInfo,
)
from voicevox_engine.model import AivmInfo, LibrarySpeaker

__all__ = ["AivmManager"]


class AivmManager:
    """
    AIVM (Aivis Voice Model) ファイルフォーマットの音声合成モデルを管理するクラス
    VOICEVOX ENGINE の LibraryManager がベースだが、AivisSpeech Engine 向けに大幅に改変されている
    """

    # AivisSpeech でサポートされているマニフェストバージョン
    SUPPORTED_MANIFEST_VERSIONS: list[str] = ["1.0"]
    # AivisSpeech でサポートされている音声合成モデルのアーキテクチャ
    SUPPORTED_MODEL_ARCHITECTURES: list[ModelArchitecture] = [
        ModelArchitecture.StyleBertVITS2,
        ModelArchitecture.StyleBertVITS2JPExtra,
    ]

    def __init__(self, installed_aivm_dir: Path):
        """
        AivmManager のコンストラクタ

        Parameters
        ----------
        installed_aivm_dir : Path
            AIVM ファイルのインストール先ディレクトリ
        """

        self.installed_aivm_dir = installed_aivm_dir
        self.installed_aivm_dir.mkdir(exist_ok=True)

        logger.info("Installed AIVM models:")
        for aivm_info in self.get_installed_aivm_infos().values():
            logger.info(f"- {aivm_info.manifest.name} ({aivm_info.manifest.uuid})")

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
        AIVM ファイルの UUID から AIVM ファイルの情報を取得する

        Parameters
        ----------
        aivm_uuid : str
            音声合成モデルの UUID (aivm_manifest.json に記載されているものと同一)

        Returns
        -------
        aivm_info : AivmInfo
            AIVM ファイルの情報
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
                                    local_style_id = self.style_id_to_local_style_id(style_id)
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
            インストール済み音声合成モデルの情報 (キー: AIVM ファイルの UUID, 値: AivmInfo)
        """

        # AIVM ファイルのインストール先ディレクトリ内に配置されている .aivm ファイルのパスを取得
        aivm_file_paths = glob.glob(str(self.installed_aivm_dir / "*.aivm"))

        # 各 AIVM ファイルごとに
        aivm_infos: dict[str, AivmInfo] = {}
        for aivm_file_path in aivm_file_paths:

            # 最低限のパスのバリデーション
            aivm_file_path = Path(aivm_file_path)
            if not aivm_file_path.exists():
                logger.warning(f"{aivm_file_path}: File not found.")
                continue
            if not aivm_file_path.is_file():
                logger.warning(f"{aivm_file_path}: Not a file.")
                continue

            # AIVM メタデータの読み込み
            try:
                with open(aivm_file_path, mode="rb") as f:
                    aivm_metadata = aivmlib.read_aivm_metadata(f)
                    aivm_manifest = aivm_metadata.manifest
            except aivmlib.AivmValidationError as e:
                logger.warning(f"{aivm_file_path}: Failed to read AIVM metadata. ({e})")
                continue

            # AIVM ファイルの UUID
            aivm_uuid = str(aivm_manifest.uuid)

            # すでに同一 UUID のファイルがインストール済みかどうかのチェック
            if aivm_uuid in aivm_infos:
                logger.info(
                    f"{aivm_file_path}: AIVM model {aivm_uuid} is already installed."
                )
                continue

            # マニフェストバージョンのバリデーション
            if aivm_manifest.manifest_version not in self.SUPPORTED_MANIFEST_VERSIONS:  # fmt: skip
                logger.warning(
                    f"{aivm_file_path}: AIVM manifest version {aivm_manifest.manifest_version} is not supported."
                )
                continue

            # 音声合成モデルのアーキテクチャのバリデーション
            if aivm_manifest.model_architecture not in self.SUPPORTED_MODEL_ARCHITECTURES:  # fmt: skip
                logger.warning(
                    f"{aivm_file_path}: Model architecture {aivm_manifest.model_architecture} is not supported."
                )
                continue

            # 仮の AivmInfo モデルを作成
            aivm_info = AivmInfo(
                # AIVM ファイルのインストール先パス
                file_path=aivm_file_path,
                # AIVM マニフェスト
                manifest=aivm_manifest,
                # 話者情報は後で追加するため、空リストを渡す
                speakers=[],
            )

            # 話者情報を LibrarySpeaker に変換し、AivmInfo.speakers に追加
            for speaker_manifest in aivm_manifest.speakers:
                speaker_uuid = str(speaker_manifest.uuid)

                # AivisSpeech Engine は日本語のみをサポートするため、日本語をサポートしない話者は除外
                ## 念のため小文字に変換してから比較
                if "ja" not in [lang.lower() for lang in speaker_manifest.supported_languages]:  # fmt: skip
                    logger.warning(f"{aivm_file_path}: Speaker {speaker_uuid} does not support Japanese. Ignoring.")  # fmt: skip
                    continue

                # スタイルごとにアセットを取得
                speaker_styles: list[SpeakerStyle] = []
                style_infos: list[StyleInfo] = []
                for style_manifest in speaker_manifest.styles:

                    # AIVM マニフェスト内の話者スタイル ID を VOICEVOX ENGINE 互換の StyleId に変換
                    style_id = self.local_style_id_to_style_id(style_manifest.local_id, speaker_uuid)  # fmt: skip

                    # SpeakerStyle の作成
                    speaker_style = SpeakerStyle(
                        # VOICEVOX ENGINE 互換のスタイル ID
                        id=style_id,
                        # スタイル名
                        name=style_manifest.name,
                        # AivisSpeech は歌唱音声合成に対応しないので talk で固定
                        type="talk",
                    )
                    speaker_styles.append(speaker_style)

                    # StyleInfo の作成
                    style_info = StyleInfo(
                        # VOICEVOX ENGINE 互換のスタイル ID
                        id=style_id,
                        # アイコン画像
                        icon=self.extract_base64_from_data_url(style_manifest.icon),
                        # 立ち絵を省略
                        ## VOICEVOX ENGINE 本家では portrait に立ち絵が入るが、AivisSpeech Engine では敢えてアイコン画像のみを設定する
                        portrait=None,
                        # ボイスサンプル
                        voice_samples=[
                            self.extract_base64_from_data_url(sample.audio)
                            for sample in style_manifest.voice_samples
                        ],
                        # 書き起こしテキスト
                        voice_sample_transcripts=[
                            sample.transcript
                            for sample in style_manifest.voice_samples
                        ],
                    )  # fmt: skip
                    style_infos.append(style_info)

                # LibrarySpeaker の作成
                aivm_info_speaker = LibrarySpeaker(
                    # 話者情報
                    speaker=Speaker(
                        # 話者 UUID
                        speaker_uuid=speaker_uuid,
                        # 話者名
                        name=speaker_manifest.name,
                        # 話者スタイル情報
                        styles=speaker_styles,
                        # 話者のバージョン
                        version=speaker_manifest.version,
                        # AivisSpeech Engine では全話者に対し常にモーフィング機能を無効化する
                        ## Style-Bert-VITS2 の仕様上音素長を一定にできず、話者ごとに発話タイミングがずれてまともに合成できないため
                        supported_features=SpeakerSupportedFeatures(
                            permitted_synthesis_morphing=SpeakerSupportPermittedSynthesisMorphing.NOTHING,
                        ),
                    ),
                    # 追加の話者情報
                    speaker_info=SpeakerInfo(
                        # 利用規約 (Markdown)
                        ## 同一 AIVM ファイル内のすべての話者は同一の利用規約を持つ
                        policy=aivm_manifest.terms_of_use,
                        # アイコン画像を Base64 エンコードして文字列化
                        ## 最初のスタイルのアイコンをこの話者全体のアイコンとして設定する
                        ## VOICEVOX ENGINE 本家では portrait に立ち絵が入るが、AivisSpeech Engine では敢えてアイコン画像を設定する
                        portrait=style_infos[0].icon,
                        # 追加の話者スタイル情報
                        style_infos=style_infos,
                    ),
                )  # fmt: skip
                aivm_info.speakers.append(aivm_info_speaker)

            # 完成した AivmInfo を UUID をキーとして追加
            aivm_infos[aivm_uuid] = aivm_info

        # 音声合成モデル名でソートしてから返す
        return dict(sorted(aivm_infos.items(), key=lambda x: x[1].manifest.name))

    def install_aivm(self, file: BinaryIO) -> None:
        """
        音声合成モデルパッケージファイル (`.aivm`) をインストールする

        Parameters
        ----------
        file : BinaryIO
            AIVM ファイルのバイナリ
        """

        # AIVM ファイルからから AIVM メタデータを取得
        try:
            aivm_metadata = aivmlib.read_aivm_metadata(file)
            aivm_manifest = aivm_metadata.manifest
        except aivmlib.AivmValidationError as e:
            raise HTTPException(
                status_code=422,
                detail=f"指定された AIVM ファイルの形式が正しくありません。({e})",
            )

        # すでに同一 UUID のファイルがインストール済みかどうかのチェック
        if str(aivm_manifest.uuid) in self.get_installed_aivm_infos():
            raise HTTPException(
                status_code=422,
                detail=f"音声合成モデル {aivm_manifest.uuid} は既にインストールされています。",
            )

        # マニフェストバージョンのバリデーション
        if aivm_manifest.manifest_version not in self.SUPPORTED_MANIFEST_VERSIONS:  # fmt: skip
            raise HTTPException(
                status_code=422,
                detail=f"AIVM マニフェストバージョン {aivm_manifest.manifest_version} には対応していません。",
            )

        # 音声合成モデルのアーキテクチャのバリデーション
        if aivm_manifest.model_architecture not in self.SUPPORTED_MODEL_ARCHITECTURES:  # fmt: skip
            raise HTTPException(
                status_code=422,
                detail=f'モデルアーキテクチャ "{aivm_manifest.model_architecture}" には対応していません。',
            )

        # AIVM ファイルをインストール
        ## 通常は重複防止のため "(AIVM ファイルの UUID).aivm" のフォーマットのファイル名でインストールされるが、
        ## 手動で .aivm ファイルをインストール先ディレクトリにコピーしても一通り動作するように考慮している
        aivm_file_path = self.installed_aivm_dir / f"{aivm_manifest.uuid}.aivm"
        with open(aivm_file_path, mode="wb") as f:
            f.write(file.read())

    def install_aivm_from_url(self, url: str) -> None:
        """
        指定された URL から音声合成モデルパッケージファイル (`.aivm`) をダウンロードしてインストールする

        Parameters
        ----------
        url : str
            AIVM ファイルの URL
        """

        # URL から AIVM ファイルをダウンロード
        try:
            response = httpx.get(
                url, headers={"User-Agent": f"AivisSpeech-Engine/{__version__}"}
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Failed to download AIVM file from {url}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"AIVM ファイルのダウンロードに失敗しました。({e})",
            )

        # ダウンロードした AIVM ファイルをインストール
        self.install_aivm(BytesIO(response.content))

    def uninstall_aivm(self, aivm_uuid: str) -> None:
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

        # AIVM ファイルをアンインストール
        ## AIVM ファイルのファイル名は必ずしも "(AIVM ファイルの UUID).aivm" になるとは限らないため、
        ## AivmInfo 内に格納されているファイルパスを使って削除する
        installed_aivm_infos[aivm_uuid].file_path.unlink()

    @staticmethod
    def local_style_id_to_style_id(local_style_id: int, speaker_uuid: str) -> StyleId:
        """
        AIVM マニフェスト内のローカルなスタイル ID を VOICEVOX ENGINE 互換のグローバルに一意な StyleId に変換する

        Parameters
        ----------
        local_style_id : int
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
        # そこで、話者の UUID とローカルなスタイル ID を組み合わせて、
        # グローバルに一意なスタイル ID (符号付き 32bit 整数) に変換する

        MAX_UUID_BITS = 27  # UUID のハッシュ値の bit 数
        UUID_BIT_MASK = (1 << MAX_UUID_BITS) - 1  # 27bit のマスク
        LOCAL_STYLE_ID_BITS = 5  # ローカルスタイル ID の bit 数
        LOCAL_STYLE_ID_MASK = (1 << LOCAL_STYLE_ID_BITS) - 1  # 5bit のマスク
        SIGN_BIT = 1 << 31  # 32bit 目の符号 bit

        if not speaker_uuid:
            raise ValueError("speaker_uuid must be a non-empty string")
        if not (0 <= local_style_id <= 31):
            raise ValueError("local_style_id must be an integer between 0 and 31")

        # UUID をハッシュ化し、27bit 整数に収める
        uuid_hash = int(hashlib.md5(speaker_uuid.encode(), usedforsecurity=False).hexdigest(), 16) & UUID_BIT_MASK  # fmt: skip
        # ローカルスタイル ID を 0 から 31 の範囲に収める
        local_style_id_masked = local_style_id & LOCAL_STYLE_ID_MASK
        # UUID のハッシュ値の下位 27bit とローカルスタイル ID の 5bit を組み合わせる
        combined_id = (uuid_hash << LOCAL_STYLE_ID_BITS) | local_style_id_masked
        # 32bit 符号付き整数として解釈するために、32bit 目が 1 の場合は正の値として扱う
        # 負の値にすると誤作動を引き起こす可能性があるため、符号ビットを反転させる
        if combined_id & SIGN_BIT:
            combined_id &= ~SIGN_BIT

        return StyleId(combined_id)

    @staticmethod
    def style_id_to_local_style_id(style_id: StyleId) -> int:
        """
        VOICEVOX ENGINE 互換のグローバルに一意な StyleId を AIVM マニフェスト内のローカルなスタイル ID に変換する

        Parameters
        ----------
        style_id : StyleId
            VOICEVOX ENGINE 互換のグローバルに一意なスタイル ID

        Returns
        -------
        local_style_id : int
            AIVM マニフェスト内のローカルなスタイル ID
        """

        # スタイル ID の下位 5 bit からローカルなスタイル ID を取り出す
        return style_id & 0x1F

    @staticmethod
    def extract_base64_from_data_url(data_url: str) -> str:
        """
        Data URL から Base64 部分のみを取り出す

        Parameters
        ----------
        data_url : str
            Data URL

        Returns
        -------
        base64 : str
            Base64 部分
        """

        # バリデーション
        if not data_url:
            raise ValueError("Data URL is empty.")
        if not data_url.startswith("data:"):
            raise ValueError("Invalid data URL format.")

        # Data URL のプレフィックスを除去して、カンマの後の Base64 エンコードされた部分を取得
        if "," in data_url:
            base64_part = data_url.split(",", 1)[1]
        else:
            raise ValueError("Invalid data URL format.")
        return base64_part
