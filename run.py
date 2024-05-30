"""VOICEVOX ENGINE の実行"""

import argparse
import multiprocessing
import os
import sys
import warnings
from io import TextIOWrapper
from pathlib import Path
from typing import TextIO, TypeVar

import uvicorn

from voicevox_engine import __version__
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.app.application import generate_app
from voicevox_engine.cancellable_engine import CancellableEngine
from voicevox_engine.core.core_initializer import MOCK_VER, initialize_cores
from voicevox_engine.engine_manifest import load_manifest
from voicevox_engine.logging import LOGGING_CONFIG, logger
from voicevox_engine.preset.Preset import PresetManager
from voicevox_engine.setting.Setting import (
    USER_SETTING_PATH,
    CorsPolicyMode,
    SettingHandler,
)
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.tts_pipeline.tts_engine import TTSEngineManager
from voicevox_engine.user_dict.user_dict import UserDictionary
from voicevox_engine.utility.path_utility import (
    engine_manifest_path,
    engine_root,
    get_save_dir,
)


def decide_boolean_from_env(env_name: str) -> bool:
    """
    環境変数からbool値を返す。

    * 環境変数が"1"ならTrueを返す
    * 環境変数が"0"か空白か存在しないならFalseを返す
    * それ以外はwarningを出してFalseを返す
    """
    env = os.getenv(env_name, default="")
    if env == "1":
        return True
    elif env == "" or env == "0":
        return False
    else:
        warnings.warn(
            f"Invalid environment variable value: {env_name}={env}",
            stacklevel=1,
        )
        return False


def set_output_log_utf8() -> None:
    """標準出力と標準エラー出力の出力形式を UTF-8 ベースに切り替える"""

    # NOTE: for 文で回せないため関数内関数で実装している
    def _prepare_utf8_stdio(stdio: TextIO | None) -> TextIO | None:
        """UTF-8 ベースの標準入出力インターフェイスを用意する"""

        CODEC = "utf-8"  # locale に依存せず UTF-8 コーデックを用いる
        ERR = "backslashreplace"  # 不正な形式のデータをバックスラッシュ付きのエスケープシーケンスに置換する

        # Python インタープリタが標準入出力へ接続されていないため設定不要とみなしそのまま返す
        if stdio is None:
            return stdio
        else:
            # 既定の `TextIOWrapper` 入出力インターフェイスを UTF-8 へ再設定して返す
            if isinstance(stdio, TextIOWrapper):
                stdio.reconfigure(encoding=CODEC)
                return stdio
            else:
                # 既定インターフェイスのバッファを全て出力しきった上で UTF-8 設定の `TextIOWrapper` を生成して返す
                stdio.flush()
                try:
                    return TextIOWrapper(stdio.buffer, encoding=CODEC, errors=ERR)
                except AttributeError:
                    # バッファへのアクセスに失敗した場合、設定変更をおこなわず返す
                    return stdio

    # NOTE:
    # `sys.std*` はコンソールがない環境だと `None` をとる (出典: https://docs.python.org/ja/3/library/sys.html#sys.__stdin__ )  # noqa: B950
    # しかし `TextIO | None` でなく `TextIO` と間違って型付けされているため、ここでは ignore している
    sys.stdout = _prepare_utf8_stdio(sys.stdout)  # type: ignore[assignment]
    sys.stderr = _prepare_utf8_stdio(sys.stderr)  # type: ignore[assignment]


T = TypeVar("T")


def select_first_not_none(candidates: list[T | None]) -> T:
    """None でない最初の値を取り出す。全て None の場合はエラーを送出する。"""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError("すべての候補値が None です")


S = TypeVar("S")


def select_first_not_none_or_none(candidates: list[S | None]) -> S | None:
    """None でない最初の値を取り出そうとし、全て None の場合は None を返す。"""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def main() -> None:
    """VOICEVOX ENGINE を実行する"""

    multiprocessing.freeze_support()

    output_log_utf8 = decide_boolean_from_env("VV_OUTPUT_LOG_UTF8")
    if output_log_utf8:
        set_output_log_utf8()

    parser = argparse.ArgumentParser(
        description="AivisSpeech Engine: AI Voice Imitation System - Text to Speech Engine"
    )
    # Uvicorn でバインドするアドレスを "localhost" にすることで IPv4 (127.0.0.1) と IPv6 ([::1]) の両方でリッスンできます.
    # これは Uvicorn のドキュメントに記載されていない挙動です; 将来のアップデートにより動作しなくなる可能性があります.
    # ref: https://github.com/VOICEVOX/voicevox_engine/pull/647#issuecomment-1540204653
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="接続を受け付けるホストアドレスです。",
    )
    parser.add_argument(
        "--port", type=int, default=10101, help="接続を受け付けるポート番号です。"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="対応している場合、GPU を使い音声合成処理を行います。",
    )
    parser.add_argument(
        "--aivisspeech_dir",
        type=Path,
        default=None,
        help="AivisSpeech のディレクトリパスです。",
    )
    # parser.add_argument(
    #     "--voicelib_dir",
    #     type=Path,
    #     default=None,
    #     action="append",
    #     help="VOICEVOX COREのディレクトリパスです。",
    # )
    # parser.add_argument(
    #     "--runtime_dir",
    #     type=Path,
    #     default=None,
    #     action="append",
    #     help="VOICEVOX COREで使用するライブラリのディレクトリパスです。",
    # )
    # parser.add_argument(
    #     "--enable_mock",
    #     action="store_true",
    #     help="VOICEVOX COREを使わずモックで音声合成を行います。",
    # )
    # parser.add_argument(
    #     "--enable_cancellable_synthesis",
    #     action="store_true",
    #     help="音声合成を途中でキャンセルできるようになります。",
    # )
    # parser.add_argument(
    #     "--init_processes",
    #     type=int,
    #     default=2,
    #     help="cancellable_synthesis機能の初期化時に生成するプロセス数です。",
    # )
    parser.add_argument(
        "--load_all_models",
        action="store_true",
        help="起動時に全ての音声合成モデルを読み込みます。",
    )

    # 引数へcpu_num_threadsの指定がなければ、環境変数をロールします。
    # 環境変数にもない場合は、Noneのままとします。
    # VV_CPU_NUM_THREADSが空文字列でなく数値でもない場合、エラー終了します。
    # parser.add_argument(
    #     "--cpu_num_threads",
    #     type=int,
    #     default=os.getenv("VV_CPU_NUM_THREADS") or None,
    #     help=(
    #         "音声合成を行うスレッド数です。指定しない場合、代わりに環境変数 VV_CPU_NUM_THREADS の値が使われます。"
    #         "VV_CPU_NUM_THREADS が空文字列でなく数値でもない場合はエラー終了します。"
    #     ),
    # )

    parser.add_argument(
        "--output_log_utf8",
        action="store_true",
        help=(
            "ログ出力を UTF-8 で行います。指定しない場合、代わりに環境変数 VV_OUTPUT_LOG_UTF8 の値が使われます。"
            "VV_OUTPUT_LOG_UTF8 の値が 1 の場合は UTF-8 で、0 または空文字、値がない場合は環境によって自動的に決定されます。"
        ),
    )

    parser.add_argument(
        "--cors_policy_mode",
        type=CorsPolicyMode,
        choices=list(CorsPolicyMode),
        default=None,
        help=(
            "CORS の許可モード。all または localapps が指定できます。all はすべてを許可します。"
            "localapps はオリジン間リソース共有ポリシーを、app://. と localhost 関連に限定します。"
            "その他のオリジンは allow_origin オプションで追加できます。デフォルトは localapps 。"
            "このオプションは --setting_file で指定される設定ファイルよりも優先されます。"
        ),
    )

    parser.add_argument(
        "--allow_origin",
        nargs="*",
        help=(
            "許可するオリジンを指定します。スペースで区切ることで複数指定できます。"
            "このオプションは --setting_file で指定される設定ファイルよりも優先されます。"
        ),
    )

    parser.add_argument(
        "--setting_file",
        type=Path,
        default=USER_SETTING_PATH,
        help="設定ファイルを指定できます。",
    )

    parser.add_argument(
        "--preset_file",
        type=Path,
        default=None,
        help=(
            "プリセットファイルを指定できます。"
            "指定がない場合、環境変数 VV_PRESET_FILE 、--aivisspeech_dir の presets.yaml 、"
            "実行ファイルのディレクトリの presets.yaml を順に探します。"
        ),
    )

    parser.add_argument(
        "--disable_mutable_api",
        action="store_true",
        help=(
            "辞書登録や設定変更など、エンジンの静的なデータを変更するAPIを無効化します。"
            "指定しない場合、代わりに環境変数 VV_DISABLE_MUTABLE_API の値が使われます。"
            "VV_DISABLE_MUTABLE_API の値が 1 の場合は無効化で、0 または空文字、値がない場合は無視されます。"
        ),
    )

    args = parser.parse_args()

    # NOTE: 型検査のため Any 値に対して明示的に型を付ける
    arg_cors_policy_mode: CorsPolicyMode | None = args.cors_policy_mode
    arg_allow_origin: list[str] | None = args.allow_origin
    arg_preset_path: Path | None = args.preset_file
    arg_disable_mutable_api: bool = args.disable_mutable_api

    if args.output_log_utf8:
        set_output_log_utf8()

    logger.info(f"AivisSpeech Engine version {__version__}")

    # Synthesis Engine
    use_gpu: bool = args.use_gpu
    voicevox_dir: Path | None = args.aivisspeech_dir
    # voicelib_dirs: list[Path] | None = args.voicelib_dir
    voicelib_dirs: list[Path] | None = None  # 常に None
    # runtime_dirs: list[Path] | None = args.runtime_dir
    runtime_dirs: list[Path] | None = None  # 常に None
    # enable_mock: bool = args.enable_mock
    enable_mock: bool = True  # 常に有効化
    # cpu_num_threads: int | None = args.cpu_num_threads
    cpu_num_threads: int | None = 4  # 常に 4
    load_all_models: bool = args.load_all_models

    # 常にモックの Core を利用する
    core_manager = initialize_cores(
        use_gpu=use_gpu,
        voicelib_dirs=voicelib_dirs,
        voicevox_dir=voicevox_dir,
        runtime_dirs=runtime_dirs,
        cpu_num_threads=cpu_num_threads,
        enable_mock=enable_mock,
        load_all_models=load_all_models,
    )
    # tts_engines = make_tts_engines_from_cores(core_manager)
    # assert len(tts_engines.versions()) != 0, "音声合成エンジンがありません。"
    # latest_core_version = tts_engines.latest_version()
    latest_core_version = MOCK_VER

    # AivmManager を初期化
    aivm_manager = AivmManager(get_save_dir() / "aivm_models")

    # StyleBertVITS2TTSEngine を通常の TTSEngine の代わりに利用
    tts_engines = TTSEngineManager()
    tts_engines.register_engine(
        StyleBertVITS2TTSEngine(aivm_manager, use_gpu, load_all_models), MOCK_VER
    )

    # Cancellable Engine
    # enable_cancellable_synthesis: bool = args.enable_cancellable_synthesis
    enable_cancellable_synthesis: bool = False  # 常に無効化
    # init_processes: int = args.init_processes
    init_processes: int = 2  # 常に2

    cancellable_engine: CancellableEngine | None = None
    if enable_cancellable_synthesis:
        cancellable_engine = CancellableEngine(
            init_processes=init_processes,
            use_gpu=use_gpu,
            voicelib_dirs=voicelib_dirs,
            voicevox_dir=voicevox_dir,
            runtime_dirs=runtime_dirs,
            cpu_num_threads=cpu_num_threads,
            enable_mock=enable_mock,
        )

    setting_loader = SettingHandler(args.setting_file)
    settings = setting_loader.load()

    # 複数方式で指定可能な場合、優先度は上から「引数」「環境変数」「設定ファイル」「デフォルト値」

    root_dir = select_first_not_none([voicevox_dir, engine_root()])

    cors_policy_mode = select_first_not_none(
        [arg_cors_policy_mode, settings.cors_policy_mode]
    )

    setting_allow_origin = None
    if settings.allow_origin is not None:
        setting_allow_origin = settings.allow_origin.split(" ")
    allow_origin = select_first_not_none_or_none(
        [arg_allow_origin, setting_allow_origin]
    )

    env_preset_path_str = os.getenv("VV_PRESET_FILE")
    if env_preset_path_str is not None and len(env_preset_path_str) != 0:
        env_preset_path = Path(env_preset_path_str)
    else:
        env_preset_path = None
    root_preset_path = root_dir / "presets.yaml"
    preset_path = select_first_not_none(
        [arg_preset_path, env_preset_path, root_preset_path]
    )
    # ファイルの存在に関わらず指定されたパスをプリセットファイルとして使用する
    preset_manager = PresetManager(preset_path)

    user_dict = UserDictionary()

    engine_manifest = load_manifest(engine_manifest_path())

    if arg_disable_mutable_api:
        disable_mutable_api = True
    else:
        disable_mutable_api = decide_boolean_from_env("VV_DISABLE_MUTABLE_API")

    # ASGI に準拠した VOICEVOX ENGINE アプリケーションを生成する
    app = generate_app(
        tts_engines,
        aivm_manager,
        core_manager,
        latest_core_version,
        setting_loader,
        preset_manager,
        user_dict,
        engine_manifest,
        cancellable_engine,
        root_dir,
        cors_policy_mode,
        allow_origin,
        disable_mutable_api=disable_mutable_api,
    )

    # VOICEVOX ENGINE サーバーを起動
    # NOTE: デフォルトは ASGI に準拠した HTTP/1.1 サーバー
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_config=LOGGING_CONFIG)
    except KeyboardInterrupt:
        print("`KeyboardInterrupt` の検出によりエンジンを停止しました。")
        pass


if __name__ == "__main__":
    main()
