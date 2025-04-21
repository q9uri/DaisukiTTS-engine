from pathlib import Path

from aivmlib.schemas.aivm_manifest import AivmManifest
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from voicevox_engine.library.model import LibrarySpeaker
from voicevox_engine.tts_pipeline.model import AccentPhrase


class AudioQuery(BaseModel):
    """音声合成用のクエリ。"""

    accent_phrases: list[AccentPhrase] = Field(title="アクセント句のリスト")
    speedScale: float = Field(
        title="全体の話速",
        description=(
            "全体の話速を 0.5 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "2.0 で 2 倍速、0.5 で 0.5 倍速になる。"
        ),
    )
    intonationScale: float = Field(
        title="全体のスタイルの強さ (「全体の抑揚」ではない点で VOICEVOX ENGINE と異なる)",
        description=(
            "話者スタイルの声色の強弱を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "値が大きいほどそのスタイルに近い抑揚がついた声になる。\n"
            "例えば話者スタイルが「うれしい」なら、値が大きいほどより嬉しそうな明るい話し方になる。\n"
            "一方、話者やスタイルによっては、数値を上げすぎると発声がおかしくなったり、棒読みで不自然な声になる場合もある。\n"
            "ちゃんと発声できる「スタイルの強さ」の上限は話者やスタイルによって異なるため、適宜調整が必要。\n"
            "全スタイルの平均であるノーマルスタイルには指定できない (値にかかわらず無視される) 。"
        ),
    )
    tempoDynamicsScale: float = Field(
        default=1.0,
        title="全体のテンポの緩急 (AivisSpeech Engine 固有のフィールド)",
        description=(
            "話す速さの緩急の強弱を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "値が大きいほどより早口で生っぽい抑揚がついた声になる。\n"
            "VOICEVOX ENGINE との互換性のため、未指定時はデフォルト値が適用される。"
        ),
    )
    pitchScale: float = Field(
        title="全体の音高",
        description=(
            "全体の音高を -0.15 ~ 0.15 の範囲で指定する (デフォルト: 0.0) 。\n"
            "値が大きいほど高い声になる。\n"
            "VOICEVOX ENGINE と異なり、この値を 0.0 から変更すると音質が劣化するため注意が必要。"
        ),
    )
    volumeScale: float = Field(
        title="全体の音量",
        description=(
            "全体の音量を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "値が大きいほど大きな声になる。"
        ),
    )
    prePhonemeLength: float = Field(title="音声の前の無音時間 (秒)")
    postPhonemeLength: float = Field(title="音声の後の無音時間 (秒)")
    pauseLength: float | None = Field(
        default=None,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間。null のときは無視される。デフォルト値は null 。",
    )
    pauseLengthScale: float = Field(
        default=1,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間（倍率）。デフォルト値は 1 。",
    )
    outputSamplingRate: int = Field(title="音声データの出力サンプリングレート")
    outputStereo: bool = Field(title="音声データをステレオ出力するか否か")
    kana: str | SkipJsonSchema[None] = Field(
        default=None,
        title="読み上げるテキスト (「読みの AquesTalk 風記法テキスト」ではない点で VOICEVOX ENGINE と異なる)",
        description=(
            "読み上げるテキストを指定する。\n"
            "VOICEVOX ENGINE では AquesTalk 風記法テキストが入る読み取り専用フィールドだが (音声合成時には無視される) 、"
            "AivisSpeech Engine では音声合成時に漢字や記号が含まれた通常の読み上げテキストも必要なため、"
            "苦肉の策で読み上げテキスト指定用のフィールドとして転用した。\n"
            "VOICEVOX ENGINE との互換性のため None や空文字列が指定された場合も動作するが、"
            "その場合はアクセント句から自動生成されたひらがな文字列が読み上げテキストになるため、不自然なイントネーションになってしまう。\n"
            "可能な限り kana に通常の読み上げテキストを指定した上で音声合成 API に渡すことを推奨する。"
        ),
    )

    def __hash__(self) -> int:
        """内容に対して一意なハッシュ値を返す。"""
        # NOTE: lru_cache がユースケースのひとつ
        items = [
            (k, tuple(v)) if isinstance(v, list) else (k, v)
            for k, v in self.__dict__.items()
        ]
        return hash(tuple(sorted(items)))


class AivmInfo(BaseModel):
    """
    AIVM (Aivis Voice Model) 仕様に準拠した音声合成モデルのメタデータ情報。

    AIVM マニフェストには、音声合成モデルに関連する全てのメタデータが含まれる。
    speakers フィールド内の話者情報は、VOICEVOX ENGINE との API 互換性のために、
    AIVM マニフェストを基に Speaker / SpeakerStyle / SpeakerInfo / StyleInfo モデルに変換したもの。
    """

    is_loaded: bool = Field(title="この音声合成モデルがロードされているかどうか")
    is_update_available: bool = Field(
        title="この音声合成モデルの新しいバージョンが AivisHub で公開されているかどうか"
    )
    latest_version: str = Field(
        title="この音声合成モデルの AivisHub で公開されている最新バージョン (AivisHub で公開されていない場合は AIVM マニフェスト記載のバージョン)"
    )
    file_path: Path = Field(title="AIVMX ファイルのインストール先パス")
    file_size: int = Field(title="AIVMX ファイルのインストールサイズ (バイト単位)")
    manifest: AivmManifest = Field(title="AIVM マニフェスト")
    speakers: list[LibrarySpeaker] = Field(
        title="話者情報のリスト (VOICEVOX ENGINE 互換)"
    )
