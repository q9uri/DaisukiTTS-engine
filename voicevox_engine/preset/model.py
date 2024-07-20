"""
プリセット機能に関して API と ENGINE 内部実装が共有するモデル（データ構造）

モデルの注意点は `voicevox_engine/model.py` の module docstring を確認すること。
"""

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from voicevox_engine.metas.Metas import StyleId


class Preset(BaseModel):
    """
    プリセット情報
    """

    id: int = Field(title="プリセット ID")
    name: str = Field(title="プリセット名")
    speaker_uuid: str = Field(title="キャラクターの UUID")
    style_id: StyleId = Field(title="スタイル ID")
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
            "一方スタイルによっては値を大きくしすぎると不自然な棒読みボイスになりがちなので、適宜調整が必要。\n"
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
    pauseLength: float | SkipJsonSchema[None] = Field(
        default=None,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間",
    )
    pauseLengthScale: float = Field(
        default=1,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間（倍率）",
    )
