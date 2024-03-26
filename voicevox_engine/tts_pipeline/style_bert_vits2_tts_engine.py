# flake8: noqa

import copy

import numpy as np
from numpy.typing import NDArray
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.tts_model import TTSModel

from ..aivm_manager import AivmManager
from ..dev.core.mock import MockCoreWrapper
from ..metas.Metas import StyleId
from ..model import AudioQuery
from ..tts_pipeline.tts_engine import TTSEngine, to_flatten_moras
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

        # PyTorch に渡すデバイス名
        self.device = "cuda" if use_gpu else "cpu"

        # 音声合成に必要な BERT モデル・トークナイザーを読み込む
        ## 一度ロードすればプロセス内でグローバルに保持される
        print("Loading BERT model and tokenizer...")
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
        print("BERT model and tokenizer loaded.")

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
            model_path=self.aivm_manager.installed_aivm_dir
            / aivm_uuid
            / "model.safetensors",
            config_path=self.aivm_manager.installed_aivm_dir
            / aivm_uuid
            / "config.json",
            style_vec_path=self.aivm_manager.installed_aivm_dir
            / aivm_uuid
            / "style_vectors.npy",
            device=self.device,
        )
        tts_model.load()
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
        """

        # モーフィング時などに同一参照の AudioQuery で複数回呼ばれる可能性があるので、元の引数の AudioQuery に破壊的変更を行わない
        query = copy.deepcopy(query)

        # 読み仮名 (カタカナのみ) のテキストを取得
        flatten_moras = to_flatten_moras(query.accent_phrases)
        text = "".join([mora.text for mora in flatten_moras]) + "。"

        # 読み上げテキストを正規化
        text = normalize_text(text)
        print(f"Normalized text: {text}")

        # スタイル ID に基づく AivmManifest を取得
        aivm_manifest = self.aivm_manager.get_aivm_manifest_from_style_id(style_id)

        # 音声合成モデルをロード (初回のみ)
        print(f"Loading TTS model for AIVM {aivm_manifest.uuid} ...")
        model = self.load_model(aivm_manifest.uuid)
        print(f"TTS model loaded for AIVM {aivm_manifest.uuid} .")

        # 音声合成を実行
        ## infer() に渡されない AudioQuery のパラメータは無視される (volumeScale のみ合成後に適用される)
        print("Running inference...")
        _, wave = model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=0,
            style="ノーマル",
            length=(1 / query.speedScale),
            pitch_scale=query.pitchScale,
            sdp_ratio=query.intonationScale,
        )
        print("Inference done.")

        # 生成した音声の音量を調整
        # wave = wave.astype(np.float32) / np.iinfo(wave.dtype).max
        # wave = wave * query.volumeScale

        return wave
