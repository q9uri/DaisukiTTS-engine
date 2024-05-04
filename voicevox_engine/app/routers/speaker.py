"""話者情報機能を提供する API Router"""

import base64
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.metas.Metas import StyleId
from voicevox_engine.model import Speaker, SpeakerInfo


def b64encode_str(s: bytes) -> str:
    return base64.b64encode(s).decode("utf-8")


def generate_speaker_router(aivm_manager: AivmManager) -> APIRouter:
    """話者情報 API Router を生成する"""
    router = APIRouter()

    @router.get("/speakers", response_model=list[Speaker], tags=["その他"])
    def speakers(
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> list[Speaker]:
        # AivisSpeech Engine では常に AivmManager から Speaker を取得する
        return aivm_manager.get_speakers()
        # speakers = metas_store.load_combined_metas(get_core(core_version))
        # return filter_speakers_and_styles(speakers, "speaker")

    @router.get("/speaker_info", response_model=SpeakerInfo, tags=["その他"])
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

    # # FIXME: この関数をどこかに切り出す
    # def _speaker_info(
    #     speaker_uuid: str,
    #     speaker_or_singer: Literal["speaker", "singer"],
    #     core_version: str | None,
    # ) -> SpeakerInfo:
    #     # エンジンに含まれる話者メタ情報は、次のディレクトリ構造に従わなければならない：
    #     # {root_dir}/
    #     #   speaker_info/
    #     #       {speaker_uuid_0}/
    #     #           policy.md
    #     #           portrait.png
    #     #           icons/
    #     #               {id_0}.png
    #     #               {id_1}.png
    #     #               ...
    #     #           portraits/
    #     #               {id_0}.png
    #     #               {id_1}.png
    #     #               ...
    #     #           voice_samples/
    #     #               {id_0}_001.wav
    #     #               {id_0}_002.wav
    #     #               {id_0}_003.wav
    #     #               {id_1}_001.wav
    #     #               ...
    #     #       {speaker_uuid_1}/
    #     #           ...

    #     # 該当話者の検索
    #     speakers = parse_obj_as(
    #         list[Speaker], json.loads(get_core(core_version).speakers)
    #     )
    #     speakers = filter_speakers_and_styles(speakers, speaker_or_singer)
    #     for i in range(len(speakers)):
    #         if speakers[i].speaker_uuid == speaker_uuid:
    #             speaker = speakers[i]
    #             break
    #     else:
    #         raise HTTPException(status_code=404, detail="該当する話者が見つかりません")

    #     try:
    #         speaker_path = root_dir / "speaker_info" / speaker_uuid
    #         # 話者情報の取得
    #         # speaker policy
    #         policy_path = speaker_path / "policy.md"
    #         policy = policy_path.read_text("utf-8")
    #         # speaker portrait
    #         portrait_path = speaker_path / "portrait.png"
    #         portrait = b64encode_str(portrait_path.read_bytes())
    #         # スタイル情報の取得
    #         style_infos = []
    #         for style in speaker.styles:
    #             id = style.id
    #             # style icon
    #             style_icon_path = speaker_path / "icons" / f"{id}.png"
    #             icon = b64encode_str(style_icon_path.read_bytes())
    #             # style portrait
    #             style_portrait_path = speaker_path / "portraits" / f"{id}.png"
    #             style_portrait = None
    #             if style_portrait_path.exists():
    #                 style_portrait = b64encode_str(style_portrait_path.read_bytes())
    #             # voice samples
    #             voice_samples = [
    #                 b64encode_str(
    #                     (
    #                         speaker_path
    #                         / "voice_samples/{}_{}.wav".format(id, str(j + 1).zfill(3))
    #                     ).read_bytes()
    #                 )
    #                 for j in range(3)
    #             ]
    #             style_infos.append(
    #                 {
    #                     "id": id,
    #                     "icon": icon,
    #                     "portrait": style_portrait,
    #                     "voice_samples": voice_samples,
    #                 }
    #             )
    #     except FileNotFoundError:
    #         import traceback

    #         traceback.print_exc()
    #         raise HTTPException(
    #             status_code=500, detail="追加情報が見つかりませんでした"
    #         )

    #     ret_data = SpeakerInfo(
    #         policy=policy,
    #         portrait=portrait,
    #         style_infos=style_infos,
    #     )
    #     return ret_data

    @router.get(
        "/singers",
        response_model=list[Speaker],
        tags=["その他"],
        summary="AivisSpeech Engine ではサポートされていない API です (常に 501 Not Implemented を返します)",
    )
    def singers(
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> list[Speaker]:
        raise HTTPException(
            status_code=501,
            detail="Singers is not supported in AivisSpeech Engine.",
        )

        # singers = metas_store.load_combined_metas(get_core(core_version))
        # return filter_speakers_and_styles(singers, "singer")

    @router.get(
        "/singer_info",
        response_model=SpeakerInfo,
        tags=["その他"],
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

    @router.post("/initialize_speaker", status_code=204, tags=["その他"])
    def initialize_speaker(
        style_id: Annotated[StyleId, Query(alias="speaker")],
        skip_reinit: Annotated[
            bool,
            Query(
                description="既に初期化済みのスタイルの再初期化をスキップするかどうか",
            ),
        ] = False,
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> Response:
        """
        指定されたスタイルを初期化します。
        実行しなくても他のAPIは使用できますが、初回実行時に時間がかかることがあります。
        """
        # core = get_core(core_version)
        # core.initialize_style_id_synthesis(style_id, skip_reinit=skip_reinit)

        # AivisSpeech Engine では常に何もせずに 204 No Content を返す
        return Response(status_code=204)

    @router.get("/is_initialized_speaker", response_model=bool, tags=["その他"])
    def is_initialized_speaker(
        style_id: Annotated[StyleId, Query(alias="speaker")],
        core_version: Annotated[str | None, Query(description="AivisSpeech Engine ではサポートされていないパラメータです (常に無視されます) 。")] = None,  # fmt: skip # noqa
    ) -> bool:
        """
        指定されたスタイルが初期化されているかどうかを返します。
        """
        # core = get_core(core_version)
        # return core.is_initialized_style_id_synthesis(style_id)

        # AivisSpeech Engine では常に True を返す
        return True

    return router
