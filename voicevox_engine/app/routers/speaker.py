"""話者情報機能を提供する API Router"""

import base64
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.metas.Metas import StyleId
from voicevox_engine.model import Speaker, SpeakerInfo
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.tts_pipeline.tts_engine import TTSEngineManager


def b64encode_str(s: bytes) -> str:
    return base64.b64encode(s).decode("utf-8")


def generate_speaker_router(
    tts_engines: TTSEngineManager,
    aivm_manager: AivmManager,
) -> APIRouter:
    """話者情報 API Router を生成する"""
    router = APIRouter(tags=["その他"])
    tts_engine = tts_engines.get_engine()

    @router.get("/speakers")
    def speakers(
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> list[Speaker]:
        # AivisSpeech Engine では常に AivmManager から Speaker を取得する
        return aivm_manager.get_speakers()
        # speakers = metas_store.load_combined_metas(core_manager.get_core(core_version))
        # return filter_speakers_and_styles(speakers, "speaker")

    @router.get("/speaker_info")
    def speaker_info(
        speaker_uuid: Annotated[str, Query(..., description="話者の UUID 。")],  # noqa
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> SpeakerInfo:
        """
        指定されたspeaker_uuidに関する情報をjson形式で返します。
        画像や音声はbase64エンコードされたものが返されます。
        """
        # AivisSpeech Engine では常に AivmManager から SpeakerInfo を取得する
        return aivm_manager.get_speaker_info(speaker_uuid)
        # return _speaker_info(
        #     speaker_uuid=speaker_uuid,
        #     speaker_or_singer="speaker",
        #     core_version=core_version,
        # )

    """
    # FIXME: この関数をどこかに切り出す
    def _speaker_info(
        speaker_uuid: str,
        speaker_or_singer: Literal["speaker", "singer"],
        core_version: str | None,
    ) -> SpeakerInfo:
        # エンジンに含まれる話者メタ情報は、次のディレクトリ構造に従わなければならない：
        # {root_dir}/
        #   speaker_info/
        #       {speaker_uuid_0}/
        #           policy.md
        #           portrait.png
        #           icons/
        #               {id_0}.png
        #               {id_1}.png
        #               ...
        #           portraits/
        #               {id_0}.png
        #               {id_1}.png
        #               ...
        #           voice_samples/
        #               {id_0}_001.wav
        #               {id_0}_002.wav
        #               {id_0}_003.wav
        #               {id_1}_001.wav
        #               ...
        #       {speaker_uuid_1}/
        #           ...

        # 該当話者を検索する
        speakers = parse_obj_as(
            list[Speaker], json.loads(core_manager.get_core(core_version).speakers)
        )
        speakers = filter_speakers_and_styles(speakers, speaker_or_singer)
        speaker = next(
            filter(lambda spk: spk.speaker_uuid == speaker_uuid, speakers), None
        )
        if speaker is None:
            raise HTTPException(status_code=404, detail="該当する話者が見つかりません")

        # 話者情報を取得する
        try:
            speaker_path = root_dir / "speaker_info" / speaker_uuid

            # speaker policy
            policy_path = speaker_path / "policy.md"
            policy = policy_path.read_text("utf-8")

            # speaker portrait
            portrait_path = speaker_path / "portrait.png"
            portrait = b64encode_str(portrait_path.read_bytes())

            # スタイル情報を取得する
            style_infos = []
            for style in speaker.styles:
                id = style.id

                # style icon
                style_icon_path = speaker_path / "icons" / f"{id}.png"
                icon = b64encode_str(style_icon_path.read_bytes())

                # style portrait
                style_portrait_path = speaker_path / "portraits" / f"{id}.png"
                style_portrait = None
                if style_portrait_path.exists():
                    style_portrait = b64encode_str(style_portrait_path.read_bytes())

                # voice samples
                voice_samples: list[str] = []
                for j in range(3):
                    num = str(j + 1).zfill(3)
                    voice_path = speaker_path / "voice_samples" / f"{id}_{num}.wav"
                    voice_samples.append(b64encode_str(voice_path.read_bytes()))

                style_infos.append(
                    {
                        "id": id,
                        "icon": icon,
                        "portrait": style_portrait,
                        "voice_samples": voice_samples,
                    }
                )
        except FileNotFoundError:
            traceback.print_exc()
            msg = "追加情報が見つかりませんでした"
            raise HTTPException(status_code=500, detail=msg)

        spk_info = SpeakerInfo(
            policy=policy, portrait=portrait, style_infos=style_infos
        )
        return spk_info
    """

    @router.get(
        "/singers",
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def singers(
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> list[Speaker]:
        raise HTTPException(
            status_code=501,
            detail="Singers is not supported in AivisSpeech Engine.",
        )
        # singers = metas_store.load_combined_metas(core_manager.get_core(core_version))
        # return filter_speakers_and_styles(singers, "singer")

    @router.get(
        "/singer_info",
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def singer_info(
        speaker_uuid: Annotated[str, Query(..., description="話者の UUID 。")],  # noqa
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> SpeakerInfo:
        # """
        # 指定されたspeaker_uuidに関する情報をjson形式で返します。
        # 画像や音声はbase64エンコードされたものが返されます。
        # """
        raise HTTPException(
            status_code=501,
            detail="Singer info is not supported in AivisSpeech Engine.",
        )
        # return _speaker_info(
        #     speaker_uuid=speaker_uuid,
        #     speaker_or_singer="singer",
        #     core_version=core_version,
        # )

    @router.post("/initialize_speaker", status_code=204)
    def initialize_speaker(
        style_id: Annotated[StyleId, Query(alias="speaker")],
        skip_reinit: Annotated[
            bool,
            Query(
                description="既に初期化済みのスタイルの再初期化をスキップするかどうか",
            ),
        ] = False,
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> None:
        """
        指定されたスタイルを初期化します。
        実行しなくても他のAPIは使用できますが、初回実行時に時間がかかることがあります。
        """
        # core = core_manager.get_core(core_version)
        # core.initialize_style_id_synthesis(style_id, skip_reinit=skip_reinit)

        # テスト用 TTSEngine の場合は何もしない
        if not isinstance(tts_engine, StyleBertVITS2TTSEngine):
            return

        # AivisSpeech Engine ではスタイル ID に対応する AivmManifest を取得後、
        # AIVM マニフェスト記載の UUID に対応する音声合成モデルをロードする
        aivm_manifest, _, _ = aivm_manager.get_aivm_manifest_from_style_id(style_id)
        tts_engine.load_model(aivm_manifest.uuid)

    @router.get("/is_initialized_speaker")
    def is_initialized_speaker(
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> bool:
        """
        指定されたスタイルが初期化されているかどうかを返します。
        """
        # core = core_manager.get_core(core_version)
        # return core.is_initialized_style_id_synthesis(style_id)

        # テスト用 TTSEngine の場合は常に True を返す
        if not isinstance(tts_engine, StyleBertVITS2TTSEngine):
            return True

        # AivisSpeech Engine ではスタイル ID に対応する AivmManifest を取得後、
        # AIVM マニフェスト記載の UUID に対応する音声合成モデルがロードされているかどうかを返す
        aivm_manifest, _, _ = aivm_manager.get_aivm_manifest_from_style_id(style_id)
        return tts_engine.is_model_loaded(aivm_manifest.uuid)

    return router
