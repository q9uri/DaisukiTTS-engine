import argparse
import asyncio
import json
import multiprocessing
import os
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO, TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Optional

import soundfile
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi import Path as FAPath
from fastapi import Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.responses import FileResponse

from voicevox_engine import __version__
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.app.dependencies import (
    check_disabled_mutable_api,
    deprecated_mutable_api,
)
from voicevox_engine.app.routers import (
    preset,
    setting,
    speaker,
    tts_pipeline,
    user_dict,
)
from voicevox_engine.cancellable_engine import CancellableEngine
from voicevox_engine.core.core_adapter import CoreAdapter
from voicevox_engine.core.core_initializer import MOCK_VER, initialize_cores
from voicevox_engine.engine_manifest.EngineManifest import EngineManifest
from voicevox_engine.engine_manifest.EngineManifestLoader import EngineManifestLoader
from voicevox_engine.logging import LOGGING_CONFIG, logger
from voicevox_engine.metas.Metas import StyleId
from voicevox_engine.metas.MetasStore import construct_lookup
from voicevox_engine.model import (
    AivmInfo,
    AivmManifest,
    AudioQuery,
    MorphableTargetInfo,
    StyleIdNotFoundError,
    SupportedDevicesInfo,
)
from voicevox_engine.morphing import (
    get_morphable_targets,
    is_synthesis_morphing_permitted,
    synthesis_morphing,
)
from voicevox_engine.morphing import (
    synthesis_morphing_parameter as _synthesis_morphing_parameter,
)
from voicevox_engine.preset.PresetManager import PresetManager
from voicevox_engine.setting.Setting import CorsPolicyMode
from voicevox_engine.setting.SettingLoader import USER_SETTING_PATH, SettingHandler
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.tts_pipeline.tts_engine import TTSEngine
from voicevox_engine.user_dict.user_dict import update_dict
from voicevox_engine.utility.path_utility import delete_file, engine_root, get_save_dir
from voicevox_engine.utility.run_utility import decide_boolean_from_env


def set_output_log_utf8() -> None:
    """
    stdout/stderrのエンコーディングをUTF-8に切り替える関数
    """
    # コンソールがない環境だとNone https://docs.python.org/ja/3/library/sys.html#sys.__stdin__
    if sys.stdout is not None:
        if isinstance(sys.stdout, TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8")
        else:
            # バッファを全て出力する
            sys.stdout.flush()
            try:
                sys.stdout = TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="backslashreplace"
                )
            except AttributeError:
                # stdout.bufferがない場合は無視
                pass
    if sys.stderr is not None:
        if isinstance(sys.stderr, TextIOWrapper):
            sys.stderr.reconfigure(encoding="utf-8")
        else:
            sys.stderr.flush()
            try:
                sys.stderr = TextIOWrapper(
                    sys.stderr.buffer, encoding="utf-8", errors="backslashreplace"
                )
            except AttributeError:
                # stderr.bufferがない場合は無視
                pass


def generate_app(
    tts_engines: dict[str, TTSEngine],
    aivm_manager: AivmManager,
    cores: dict[str, CoreAdapter],
    latest_core_version: str,
    setting_loader: SettingHandler,
    preset_manager: PresetManager,
    cancellable_engine: CancellableEngine | None = None,
    root_dir: Optional[Path] = None,
    cors_policy_mode: CorsPolicyMode = CorsPolicyMode.localapps,
    allow_origin: Optional[list[str]] = None,
    disable_mutable_api: bool = False,
) -> FastAPI:
    if root_dir is None:
        root_dir = engine_root()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        update_dict()
        yield

    app = FastAPI(
        title="AivisSpeech Engine",
        description="AivisSpeech の音声合成エンジンです。",
        version=__version__,
        lifespan=lifespan,
        # OpenAPI Generator が自動生成するコードとの互換性が壊れるため、リクエストとレスポンスで Pydantic スキーマを分離しないようにする
        # ref: https://fastapi.tiangolo.com/how-to/separate-openapi-schemas/
        separate_input_output_schemas=False,
    )

    # 未処理の例外が発生するとCORSMiddlewareが適用されない問題に対するワークアラウンド
    # ref: https://github.com/VOICEVOX/voicevox_engine/issues/91
    async def global_execution_handler(request: Request, exc: Exception) -> Response:
        logger.error("Internal Server Error occurred.", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content="Internal Server Error",
        )

    app.add_middleware(ServerErrorMiddleware, handler=global_execution_handler)

    # CORS用のヘッダを生成するミドルウェア
    localhost_regex = "^https?://(localhost|127\\.0\\.0\\.1)(:[0-9]+)?$"
    compiled_localhost_regex = re.compile(localhost_regex)
    allowed_origins = ["*"]
    if cors_policy_mode == "localapps":
        allowed_origins = ["app://."]
        if allow_origin is not None:
            allowed_origins += allow_origin
            if "*" in allow_origin:
                logger.warning(
                    'Deprecated use of argument "*" in allow_origin. '
                    'Use option "--cors_policy_mod all" instead. See "--help" for more.'
                )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_origin_regex=localhost_regex,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 許可されていないOriginを遮断するミドルウェア
    @app.middleware("http")
    async def block_origin_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response | JSONResponse:
        isValidOrigin: bool = False
        if "Origin" not in request.headers:  # Originのない純粋なリクエストの場合
            isValidOrigin = True
        elif "*" in allowed_origins:  # すべてを許可する設定の場合
            isValidOrigin = True
        elif request.headers["Origin"] in allowed_origins:  # Originが許可されている場合
            isValidOrigin = True
        elif compiled_localhost_regex.fullmatch(
            request.headers["Origin"]
        ):  # localhostの場合
            isValidOrigin = True

        if isValidOrigin:
            return await call_next(request)
        else:
            return JSONResponse(
                status_code=403, content={"detail": "Origin not allowed"}
            )

    if disable_mutable_api:
        deprecated_mutable_api.enable = False

    engine_manifest_data = EngineManifestLoader(
        engine_root() / "engine_manifest.json",
        engine_root(),
    ).load_manifest()

    # metas_store = MetasStore(root_dir / "speaker_info")

    setting_ui_template = Jinja2Templates(
        directory=engine_root() / "ui_template",
        variable_start_string="<JINJA_PRE>",
        variable_end_string="<JINJA_POST>",
    )

    # キャッシュを有効化
    # モジュール側でlru_cacheを指定するとキャッシュを制御しにくいため、HTTPサーバ側で指定する
    # TODO: キャッシュを管理するモジュール側API・HTTP側APIを用意する
    synthesis_morphing_parameter = lru_cache(maxsize=4)(_synthesis_morphing_parameter)

    # @app.on_event("startup")
    # async def start_catch_disconnection():
    #     if cancellable_engine is not None:
    #         loop = asyncio.get_event_loop()
    #         _ = loop.create_task(cancellable_engine.catch_disconnection())

    def get_engine(core_version: Optional[str]) -> TTSEngine:
        if core_version is None:
            return tts_engines[latest_core_version]
        if core_version in tts_engines:
            return tts_engines[core_version]
        raise HTTPException(status_code=422, detail="不明なバージョンです")

    def get_core(core_version: Optional[str]) -> CoreAdapter:
        """指定したバージョンのコアを取得する"""
        if core_version is None:
            return cores[latest_core_version]
        if core_version in cores:
            return cores[core_version]
        raise HTTPException(status_code=422, detail="不明なバージョンです")

    app.include_router(
        tts_pipeline.generate_router(
            get_engine, get_core, preset_manager, cancellable_engine
        )
    )

    @app.post(
        "/morphable_targets",
        response_model=list[dict[str, MorphableTargetInfo]],
        tags=["音声合成"],
        summary="指定したスタイルに対してエンジン内の話者がモーフィングが可能か判定する",
    )
    def morphable_targets(
        base_style_ids: list[StyleId],
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> list[dict[str, MorphableTargetInfo]]:
        """
        指定されたベーススタイルに対してエンジン内の各話者がモーフィング機能を利用可能か返します。
        モーフィングの許可/禁止は`/speakers`の`speaker.supported_features.synthesis_morphing`に記載されています。
        プロパティが存在しない場合は、モーフィングが許可されているとみなします。
        返り値のスタイルIDはstring型なので注意。
        """
        # core = get_core(core_version)

        try:
            # speakers = metas_store.load_combined_metas(core=core)
            speakers = aivm_manager.get_speakers()
            morphable_targets = get_morphable_targets(
                speakers=speakers, base_style_ids=base_style_ids
            )
            # jsonはint型のキーを持てないので、string型に変換する
            return [
                {str(k): v for k, v in morphable_target.items()}
                for morphable_target in morphable_targets
            ]
        except StyleIdNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=f"該当するスタイル(style_id={e.style_id})が見つかりません",
            )

    @app.post(
        "/synthesis_morphing",
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["音声合成"],
        summary="2種類のスタイルでモーフィングした音声を合成する",
    )
    def _synthesis_morphing(
        query: AudioQuery,
        base_style_id: StyleId = Query(alias="base_speaker"),  # noqa: B008
        target_style_id: StyleId = Query(alias="target_speaker"),  # noqa: B008
        morph_rate: float = Query(..., ge=0.0, le=1.0),  # noqa: B008
        core_version: str | None = None,
    ) -> FileResponse:
        """
        指定された2種類のスタイルで音声を合成、指定した割合でモーフィングした音声を得ます。
        モーフィングの割合は`morph_rate`で指定でき、0.0でベースのスタイル、1.0でターゲットのスタイルに近づきます。
        """
        engine = get_engine(core_version)
        core = get_core(core_version)

        try:
            # speakers = metas_store.load_combined_metas(core=core)
            speakers = aivm_manager.get_speakers()
            speaker_lookup = construct_lookup(speakers=speakers)
            is_permitted = is_synthesis_morphing_permitted(
                speaker_lookup, base_style_id, target_style_id
            )
            if not is_permitted:
                raise HTTPException(
                    status_code=400,
                    detail="指定されたスタイルペアでのモーフィングはできません",
                )
        except StyleIdNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=f"該当するスタイル(style_id={e.style_id})が見つかりません",
            )

        # 生成したパラメータはキャッシュされる
        morph_param = synthesis_morphing_parameter(
            engine=engine,
            core=core,
            query=query,
            base_style_id=base_style_id,
            target_style_id=target_style_id,
        )

        morph_wave = synthesis_morphing(
            morph_param=morph_param,
            morph_rate=morph_rate,
            output_fs=query.outputSamplingRate,
            output_stereo=query.outputStereo,
        )

        with NamedTemporaryFile(delete=False) as f:
            soundfile.write(
                file=f,
                data=morph_wave,
                samplerate=query.outputSamplingRate,
                format="WAV",
            )

        return FileResponse(
            f.name,
            media_type="audio/wav",
            background=BackgroundTask(delete_file, f.name),
        )

    app.include_router(preset.generate_router(preset_manager))

    @app.get("/version", tags=["その他"])
    async def version() -> str:
        return __version__

    @app.get("/core_versions", response_model=list[str], tags=["その他"])
    async def core_versions() -> Response:
        return Response(
            content=json.dumps(list(cores.keys())),
            media_type="application/json",
        )

    app.include_router(speaker.generate_router(aivm_manager))

    @app.get(
        "/aivm_models",
        response_model=dict[str, AivmInfo],
        response_description="インストールした音声合成モデルの情報",
        tags=["音声合成モデル管理"],
    )
    def get_installed_aivm_infos() -> dict[str, AivmInfo]:
        """
        インストールした音声合成モデルの情報を返します。
        """
        return aivm_manager.get_installed_aivm_infos()

    @app.post(
        "/aivm_models/{aivm_uuid}",
        status_code=204,
        tags=["音声合成モデル管理"],
        dependencies=[Depends(check_disabled_mutable_api)],
    )
    async def install_aivm(
        aivm_uuid: Annotated[str, FAPath(description="音声合成モデルの UUID")],
        request: Request,
    ) -> Response:
        """
        音声合成モデルをインストールします。
        音声合成モデルパッケージファイル (`.aivm`) をリクエストボディとして送信してください。
        """
        archive = BytesIO(await request.body())
        await asyncio.to_thread(aivm_manager.install_aivm, aivm_uuid, archive)
        return Response(status_code=204)

    @app.delete(
        "/aivm_models/{aivm_uuid}",
        status_code=204,
        tags=["音声合成モデル管理"],
        dependencies=[Depends(check_disabled_mutable_api)],
    )
    def uninstall_aivm(
        aivm_uuid: Annotated[str, FAPath(description="音声合成モデルの UUID")]
    ) -> Response:
        """
        音声合成モデルをアンインストールします。
        """
        aivm_manager.uninstall_aivm(aivm_uuid)
        return Response(status_code=204)

    @app.get(
        "/aivm_models/{aivm_uuid}/manifest",
        response_model=AivmManifest,
        tags=["音声合成モデル管理"],
    )
    def get_aivm_manifest(
        aivm_uuid: Annotated[str, FAPath(description="音声合成モデルの UUID")]
    ) -> AivmManifest:
        """
        音声合成モデルの AIVM マニフェストを返します。
        """
        return aivm_manager.get_aivm_manifest(aivm_uuid)

    @app.post("/initialize_speaker", status_code=204, tags=["その他"])
    def initialize_speaker(
        style_id: StyleId = Query(alias="speaker"),  # noqa: B008
        skip_reinit: bool = Query(  # noqa: B008
            default=False,
            description="既に初期化済みのスタイルの再初期化をスキップするかどうか",
        ),
        core_version: str | None = None,
    ) -> Response:
        """
        指定されたスタイルを初期化します。
        実行しなくても他のAPIは使用できますが、初回実行時に時間がかかることがあります。
        """
        core = get_core(core_version)
        core.initialize_style_id_synthesis(style_id, skip_reinit=skip_reinit)
        return Response(status_code=204)

    @app.get("/is_initialized_speaker", response_model=bool, tags=["その他"])
    def is_initialized_speaker(
        style_id: StyleId = Query(alias="speaker"),  # noqa: B008
        core_version: str | None = None,
    ) -> bool:
        """
        指定されたスタイルが初期化されているかどうかを返します。
        """
        core = get_core(core_version)
        return core.is_initialized_style_id_synthesis(style_id)

    app.include_router(user_dict.generate_router())

    @app.get("/supported_devices", response_model=SupportedDevicesInfo, tags=["その他"])
    def supported_devices(
        core_version: str | None = None,
    ) -> Response:
        supported_devices = get_core(core_version).supported_devices
        if supported_devices is None:
            raise HTTPException(status_code=422, detail="非対応の機能です。")
        return Response(
            content=supported_devices,
            media_type="application/json",
        )

    @app.get("/engine_manifest", response_model=EngineManifest, tags=["その他"])
    async def engine_manifest() -> EngineManifest:
        return engine_manifest_data

    app.include_router(
        setting.generate_router(
            setting_loader, engine_manifest_data, setting_ui_template
        )
    )

    return app


def main() -> None:
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
        "--use_gpu", action="store_true", help="GPUを使って音声合成するようになります。"
    )
    parser.add_argument(
        "--aivisspeech_dir",
        type=Path,
        default=None,
        help="AivisSpeechのディレクトリパスです。",
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
            "ログ出力をUTF-8でおこないます。指定しない場合、代わりに環境変数 VV_OUTPUT_LOG_UTF8 の値が使われます。"
            "VV_OUTPUT_LOG_UTF8 の値が1の場合はUTF-8で、0または空文字、値がない場合は環境によって自動的に決定されます。"
        ),
    )

    parser.add_argument(
        "--cors_policy_mode",
        type=CorsPolicyMode,
        choices=list(CorsPolicyMode),
        default=None,
        help=(
            "CORSの許可モード。allまたはlocalappsが指定できます。allはすべてを許可します。"
            "localappsはオリジン間リソース共有ポリシーを、app://.とlocalhost関連に限定します。"
            "その他のオリジンはallow_originオプションで追加できます。デフォルトはlocalapps。"
            "このオプションは--setting_fileで指定される設定ファイルよりも優先されます。"
        ),
    )

    parser.add_argument(
        "--allow_origin",
        nargs="*",
        help=(
            "許可するオリジンを指定します。スペースで区切ることで複数指定できます。"
            "このオプションは--setting_fileで指定される設定ファイルよりも優先されます。"
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
            "指定がない場合、環境変数 VV_PRESET_FILE、--aivisspeech_dirのpresets.yaml、"
            "実行ファイルのディレクトリのpresets.yamlを順に探します。"
        ),
    )

    parser.add_argument(
        "--disable_mutable_api",
        action="store_true",
        help=(
            "辞書登録や設定変更など、エンジンの静的なデータを変更するAPIを無効化します。"
            "指定しない場合、代わりに環境変数 VV_DISABLE_MUTABLE_API の値が使われます。"
            "VV_DISABLE_MUTABLE_API の値が1の場合は無効化で、0または空文字、値がない場合は無視されます。"
        ),
    )

    args = parser.parse_args()

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
    cores = initialize_cores(
        use_gpu=use_gpu,
        voicelib_dirs=voicelib_dirs,
        voicevox_dir=voicevox_dir,
        runtime_dirs=runtime_dirs,
        cpu_num_threads=cpu_num_threads,
        enable_mock=enable_mock,
        load_all_models=load_all_models,
    )
    # tts_engines = make_tts_engines_from_cores(cores)
    # assert len(tts_engines) != 0, "音声合成エンジンがありません。"
    # latest_core_version = get_latest_core_version(versions=list(tts_engines.keys()))
    latest_core_version = MOCK_VER

    # AivmManager を初期化
    aivm_manager = AivmManager(get_save_dir() / "installed_aivm")

    # StyleBertVITS2TTSEngine を TTSEngine の代わりに利用
    tts_engines: dict[str, TTSEngine] = {
        MOCK_VER: StyleBertVITS2TTSEngine(aivm_manager, use_gpu, load_all_models)
    }

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

    root_dir: Path | None = voicevox_dir
    if root_dir is None:
        root_dir = engine_root()

    setting_loader = SettingHandler(args.setting_file)

    settings = setting_loader.load()

    cors_policy_mode: CorsPolicyMode | None = args.cors_policy_mode
    if cors_policy_mode is None:
        cors_policy_mode = settings.cors_policy_mode

    allow_origin = None
    if args.allow_origin is not None:
        allow_origin = args.allow_origin
    elif settings.allow_origin is not None:
        allow_origin = settings.allow_origin.split(" ")

    # Preset Manager
    # preset_pathの優先順: 引数、環境変数、voicevox_dir、実行ファイルのディレクトリ
    # ファイルの存在に関わらず、優先順で最初に指定されたパスをプリセットファイルとして使用する
    preset_path: Path | None = args.preset_file
    if preset_path is None:
        # 引数 --preset_file の指定がない場合
        env_preset_path = os.getenv("VV_PRESET_FILE")
        if env_preset_path is not None and len(env_preset_path) != 0:
            # 環境変数 VV_PRESET_FILE の指定がある場合
            preset_path = Path(env_preset_path)
        else:
            # 環境変数 VV_PRESET_FILE の指定がない場合
            preset_path = root_dir / "presets.yaml"

    preset_manager = PresetManager(
        preset_path=preset_path,
    )

    disable_mutable_api: bool = args.disable_mutable_api | decide_boolean_from_env(
        "VV_DISABLE_MUTABLE_API"
    )

    uvicorn.run(
        generate_app(
            tts_engines,
            aivm_manager,
            cores,
            latest_core_version,
            setting_loader,
            preset_manager=preset_manager,
            cancellable_engine=cancellable_engine,
            root_dir=root_dir,
            cors_policy_mode=cors_policy_mode,
            allow_origin=allow_origin,
            disable_mutable_api=disable_mutable_api,
        ),  # type: ignore
        host=args.host,
        port=args.port,
        log_config=LOGGING_CONFIG,
    )


if __name__ == "__main__":
    main()
