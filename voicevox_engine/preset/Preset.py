from pydantic import BaseModel, Field

from voicevox_engine.metas.Metas import StyleId


class Preset(BaseModel):
    """
    プリセット情報
    """

    id: int = Field(title="プリセット ID")
    name: str = Field(title="プリセット名")
    speaker_uuid: str = Field(title="話者の UUID")
    style_id: StyleId = Field(title="スタイル ID")
    styleStrengthScale: float = Field(
        default=5.0,
        title="全体のスタイルの強さ (AivisSpeech Engine 固有のフィールド)",
        description=(
            "指定された話者のスタイルをどの程度全体に反映するかを指定する。\n"
            "例えばスタイルが Happy ならば、この値を大きくするほど全体の話し方が明るくなる。\n"
            "一方値を大きくしすぎると発声がおかしくなりがちなので、適宜調整が必要。\n"
            "VOICEVOX ENGINE との互換性のため、未指定時はデフォルト値が適用される。"
        ),
    )
    intonationScale: float = Field(
        title="全体のテンポの緩急 (抑揚設定ではない点で VOICEVOX ENGINE と異なる)"
    )
    speedScale: float = Field(title="全体の話速")
    pitchScale: float = Field(title="全体の音高")
    volumeScale: float = Field(title="全体の音量")
    prePhonemeLength: float = Field(title="音声の前の無音時間")
    postPhonemeLength: float = Field(title="音声の後の無音時間")
