# flake8: noqa

import copy

import jaconv
import numpy as np
import torch
from numpy.typing import NDArray
from style_bert_vits2.constants import DEFAULT_SDP_RATIO, Languages
from style_bert_vits2.logging import logger as style_bert_vits2_logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.tts_model import TTSModel

from ..aivm_manager import AivmManager
from ..dev.core.mock import MockCoreWrapper
from ..logging import logger
from ..metas.Metas import StyleId
from ..model import AudioQuery
from ..tts_pipeline.tts_engine import (
    TTSEngine,
    raw_wave_to_output_wave,
    to_flatten_moras,
)
from ..utility.path_utility import get_save_dir


class StyleBertVITS2TTSEngine(TTSEngine):
    """Style-Bert-VITS2 TTS Engine"""

    # BERT モデルのキャッシュディレクトリ
    BERT_MODEL_CACHE_DIR = get_save_dir() / "bert_model_caches"

    def __init__(
        self,
        aivm_manager: AivmManager,
        use_gpu: bool = False,
        load_all_models: bool = False,
    ) -> None:
        self.aivm_manager = aivm_manager
        self.use_gpu = use_gpu
        self.load_all_models = load_all_models

        # ロード済みモデルのキャッシュ
        self.tts_models: dict[str, TTSModel] = {}

        # PyTorch での推論に利用するデバイスを選択
        if use_gpu is True:
            if torch.backends.cuda.is_built() and torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using GPU (NVIDIA CUDA) for inference.")
            # Mac であれば基本 Apple MPS (Metal Performance Shaders) が利用できる
            elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using GPU (Apple MPS) for inference.")
            # それ以外の環境では CPU にフォールバック
            else:
                logger.warning("GPU is not available. Using CPU instead.")
                self.device = "cpu"
        else:
            self.device = "cpu"
            logger.info("Using CPU for inference.")

        # Style-Bert-VITS2 本体のロガーを抑制
        style_bert_vits2_logger.remove()

        # 音声合成に必要な BERT モデル・トークナイザーを読み込む
        ## 一度ロードすればプロセス内でグローバルに保持される
        logger.info("Loading BERT model and tokenizer...")
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
        logger.info("BERT model and tokenizer loaded.")

        # 継承元の TTSEngine の __init__ を呼び出す
        # VOICEVOX CORE の通常の CoreWrapper の代わりに MockCoreWrapper を利用する
        super().__init__(MockCoreWrapper())

    def load_model(self, aivm_uuid: str) -> TTSModel:
        """
        Style-Bert-VITS2 の音声合成モデルをロードする
        StyleBertVITS2TTSEngine の初期化時に use_gpu=True が指定されている場合、モデルは GPU にロードされる

        Parameters
        ----------
        aivm_uuid : str
            AIVM の UUID

        Returns
        -------
        TTSModel
            ロード済みの TTSModel インスタンス (キャッシュ済みの場合はそのまま返す)
        """

        # 既に読み込まれている場合はそのまま返す
        if aivm_uuid in self.tts_models:
            return self.tts_models[aivm_uuid]

        # モデルをロードする
        tts_model = TTSModel(
            model_path=self.aivm_manager.installed_aivm_dir / aivm_uuid / "model.safetensors",
            config_path=self.aivm_manager.installed_aivm_dir / aivm_uuid / "config.json",
            style_vec_path=self.aivm_manager.installed_aivm_dir / aivm_uuid / "style_vectors.npy",
            device=self.device,
        )  # fmt: skip
        logger.info("Loading model...")
        tts_model.load()
        logger.info(f"Model loaded. UUID: {aivm_uuid}")

        self.tts_models[aivm_uuid] = tts_model
        return tts_model

    def synthesize_wave(
        self,
        query: AudioQuery,
        style_id: StyleId,
        enable_interrogative_upspeak: bool = True,
    ) -> NDArray[np.float32]:
        """
        音声合成用のクエリに含まれる読み仮名に基づいて Style-Bert-VITS2 で音声波形を生成する
        継承元の TTSEngine.synthesize_wave() をオーバーライドし、Style-Bert-VITS2 用の音声合成処理に差し替えている

        Parameters
        ----------
        query : AudioQuery
            音声合成用のクエリ
        style_id : StyleId
            スタイル ID
        enable_interrogative_upspeak : bool, optional
            疑問文の場合に抑揚を上げるかどうか (VOICEVOX ENGINE との互換性維持のためのパラメータ)

        Returns
        -------
        NDArray[np.float32]
            生成された音声波形 (float32 型)
        """

        # モーフィング時などに同一参照の AudioQuery で複数回呼ばれる可能性があるので、元の引数の AudioQuery に破壊的変更を行わない
        query = copy.deepcopy(query)

        # 読み仮名 (カタカナのみ) のテキストを取得
        ## ひらがなの方がまだ抑揚の棒読み度がマシになるため、カタカナをひらがなに変換している
        flatten_moras = to_flatten_moras(query.accent_phrases)
        text = "".join([mora.text for mora in flatten_moras]) + "。"
        text = jaconv.kata2hira(text)

        # もし AudioQuery.kana に漢字混じりの通常の文章が指定されている場合はそれを使う (AivisSpeech 独自仕様)
        ## VOICEVOX ENGINE では AudioQuery.kana は読み取り専用パラメータだが、AivisSpeech Engine では
        ## 音声合成 API にアクセント句だけでなく通常の読み上げテキストを直接渡すためのパラメータとして利用している
        ## 通常の読み上げテキストの方が遥かに抑揚が自然になるため、読み仮名のみの読み上げテキストよりも優先される
        ## VOICEVOX ENGINE との互換性維持のための苦肉の策で、基本可能な限り AudioQuery.kana に読み上げテキストを指定すべき
        if query.kana is not None and query.kana != "":
            text = query.kana

        # スタイル ID と一致する AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle を取得
        result = self.aivm_manager.get_aivm_manifest_from_style_id(style_id)
        aivm_manifest = result[0]
        aivm_manifest_speaker = result[1]
        aivm_manifest_speaker_style = result[2]

        # 音声合成モデルをロード (初回のみ)
        model = self.load_model(aivm_manifest.uuid)
        logger.info(f"Model: {aivm_manifest.name} / Version {aivm_manifest.version}")  # fmt: skip
        logger.info(f"Speaker: {aivm_manifest_speaker.name} / Style: {aivm_manifest_speaker_style.name} / Version {aivm_manifest_speaker.version}")  # fmt: skip

        # ローカルな話者 ID・スタイル ID を取得
        ## 現在の Style-Bert-VITS2 の API ではスタイル ID ではなくスタイル名を指定する必要があるため、
        ## 別途 local_style_id に対応するスタイル名をハイパーパラメータから取得している
        ## AIVM マニフェスト記載のスタイル名とハイパーパラメータのスタイル名は一致していない可能性があり
        ## そのまま指定できないため、AIVM マニフェストとハイパーパラメータで共通のスタイル ID からスタイル名を取得する
        local_speaker_id: int = aivm_manifest_speaker.id
        local_style_id: int = aivm_manifest_speaker_style.id
        local_style_name: str | None = None
        for hps_style_name, hps_style_id in model.hyper_parameters.data.style2id.items():  # fmt: skip
            if hps_style_id == local_style_id:
                local_style_name = hps_style_name
                break
        if local_style_name is None:
            raise ValueError(f"Style ID {local_style_id} not found in hyper parameters.")  # fmt: skip

        # 話速を指定
        ## ref: https://github.com/litagin02/Style-Bert-VITS2/blob/2.4.1/server_editor.py#L314
        length = 1 / query.speedScale
        # ピッチを指定 (0.0 以外を指定すると音質が劣化する)
        ## pitchScale の基準は 0.0 (-1 ~ 1) なので、1.0 を基準とした 0 ~ 2 の範囲に変換する
        pitch_scale = 1.0 + query.pitchScale
        # SDP Ratio を「テンポの緩急」の比率として指定
        ## VOICEVOX では「抑揚」の比率だが、AivisSpeech では声のテンポの緩急を指定する値としている
        ## Style-Bert-VITS2 にも一応「抑揚」パラメータはあるが、pyworld で変換している関係で音質が明確に劣化する上、あまり効果がない
        ## intonationScale の基準は 1.0 (0 ~ 2) なので、DEFAULT_SDP_RATIO を基準とした 0 ~ 1 の範囲に変換する
        if 0.0 <= query.intonationScale <= 1.0:
            sdp_ratio = query.intonationScale * DEFAULT_SDP_RATIO
        elif 1.0 < query.intonationScale <= 2.0:
            sdp_ratio = DEFAULT_SDP_RATIO + (query.intonationScale - 1.0) * 0.8 / 1.0
        else:
            sdp_ratio = DEFAULT_SDP_RATIO

        # 音声合成を実行
        ## infer() に渡されない AudioQuery のパラメータは無視される (volumeScale のみ合成後に適用される)
        ## 出力音声は int16 型の NDArray で返される
        logger.info("Running inference...")
        logger.info(f"Text: {text}")
        logger.info(f"Speed:     {length:.3f} (Query: {query.speedScale})")
        logger.info(f"Pitch:     {pitch_scale:.3f} (Query: {query.pitchScale})")
        logger.info(f"SDP Ratio: {sdp_ratio:.3f} (Query: {query.intonationScale})")
        sample_rate, raw_wave = model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=local_speaker_id,
            style=local_style_name,
            length=length,
            pitch_scale=pitch_scale,
            sdp_ratio=sdp_ratio,
            # AivisSpeech Engine ではテキストの改行ごとの分割生成を行わない (エディタ側の機能と競合するため)
            line_split=False,
        )
        logger.info("Inference done.")

        # VOICEVOX CORE は float32 型の音声波形を返すため、int16 から float32 に変換して VOICEVOX CORE に合わせる
        ## float32 に変換する際に -1.0 ~ 1.0 の範囲に正規化する
        raw_wave = raw_wave.astype(np.float32) / 32768.0

        # 生成した音声の音量調整/サンプルレート変更/ステレオ化を行ってから返す
        wave = raw_wave_to_output_wave(query, raw_wave, sample_rate)
        return wave
