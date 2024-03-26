# flake8: noqa

import copy
from logging import getLogger
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyopenjtalk import tts
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models

from ..dev.core.mock import MockCoreWrapper
from ..metas.Metas import StyleId
from ..model import AudioQuery
from ..tts_pipeline.tts_engine import TTSEngine, to_flatten_moras
from ..utility.path_utility import get_save_dir


class StyleBertVITS2TTSEngine(TTSEngine):
    """Style-Bert-VITS2 TTS Engine"""

    # BERT モデルのキャッシュディレクトリ
    BERT_MODEL_CACHE_DIR = get_save_dir() / "bert_model_caches"

    def __init__(self, use_gpu: bool = False, load_all_models: bool = False) -> None:
        self.use_gpu = use_gpu
        self.load_all_models = load_all_models

        # 継承元の TTSEngine の __init__ を呼び出す
        # VOICEVOX CORE の通常の CoreWrapper の代わりに MockCoreWrapper を利用する
        super().__init__(MockCoreWrapper())

        # 音声合成に必要な BERT モデル・トークナイザーを読み込む
        ## 一度ロードすればプロセス内でグローバルに保持される
        bert_models.load_model(
            language=Languages.JP,
            pretrained_model_name_or_path="ku-nlp/deberta-v2-large-japanese-char-wwm",
            cache_dir=str(self.BERT_MODEL_CACHE_DIR),
        )
        bert_models.load_tokenizer(
            language=Languages.JP,
            pretrained_model_name_or_path="ku-nlp/deberta-v2-large-japanese-char-wwm",
            cache_dir=str(self.BERT_MODEL_CACHE_DIR),
        )

    def synthesize_wave(
        self,
        query: AudioQuery,
        style_id: StyleId,
        enable_interrogative_upspeak: bool = True,
    ) -> NDArray[np.float32]:
        """音声合成用のクエリに含まれる読み仮名に基づいてOpenJTalkで音声波形を生成する"""
        # モーフィング時などに同一参照のqueryで複数回呼ばれる可能性があるので、元の引数のqueryに破壊的変更を行わない
        query = copy.deepcopy(query)

        # recall text in katakana
        flatten_moras = to_flatten_moras(query.accent_phrases)
        kana_text = "".join([mora.text for mora in flatten_moras])

        wave = self.forward(kana_text)

        # volume
        wave *= query.volumeScale

        return wave

    def forward(self, text: str, **kwargs: dict[str, Any]) -> NDArray[np.float32]:
        """
        forward tts via pyopenjtalk.tts()
        参照→TTSEngine のdocstring [Mock]

        Parameters
        ----------
        text : str
            入力文字列（例：読み上げたい文章をカタカナにした文字列、等）

        Returns
        -------
        wave [NDArray[np.float32]]
            音声波形データをNumPy配列で返します

        Note
        -------
        ここで行う音声合成では、調声（ピッチ等）を反映しない

        # pyopenjtalk.tts()の出力仕様
        dtype=np.float64, 16 bit, mono 48000 Hz
        """
        logger = getLogger("uvicorn")  # FastAPI / Uvicorn 内からの利用のため
        logger.info("[Mock] input text: %s" % text)
        wave, _ = tts(text)
        wave /= 2**15
        return wave.astype(np.float32)
