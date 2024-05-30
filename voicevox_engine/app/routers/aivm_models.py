"""音声合成モデル管理機能を提供する API Router"""

import asyncio
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.model import AivmInfo, AivmManifest

from ..dependencies import check_disabled_mutable_api


def generate_aivm_models_router(aivm_manager: AivmManager) -> APIRouter:
    """音声合成モデル管理 API Router を生成する"""
    router = APIRouter()

    @router.get(
        "/aivm_models",
        response_description="インストールした音声合成モデルの情報",
        tags=["音声合成モデル管理"],
    )
    def get_installed_aivm_infos() -> dict[str, AivmInfo]:
        """
        インストールした音声合成モデルの情報を返します。
        """
        return aivm_manager.get_installed_aivm_infos()

    @router.post(
        "/aivm_models/{aivm_uuid}",
        status_code=204,
        tags=["音声合成モデル管理"],
        dependencies=[Depends(check_disabled_mutable_api)],
    )
    async def install_aivm(
        aivm_uuid: Annotated[str, Path(description="音声合成モデルの UUID")],
        request: Request,
    ) -> None:
        """
        音声合成モデルをインストールします。
        音声合成モデルパッケージファイル (`.aivm`) をリクエストボディとして送信してください。
        """
        archive = BytesIO(await request.body())
        await asyncio.to_thread(aivm_manager.install_aivm, aivm_uuid, archive)

    @router.delete(
        "/aivm_models/{aivm_uuid}",
        status_code=204,
        tags=["音声合成モデル管理"],
        dependencies=[Depends(check_disabled_mutable_api)],
    )
    def uninstall_aivm(
        aivm_uuid: Annotated[str, Path(description="音声合成モデルの UUID")]
    ) -> None:
        """
        音声合成モデルをアンインストールします。
        """
        aivm_manager.uninstall_aivm(aivm_uuid)

    @router.get(
        "/aivm_models/{aivm_uuid}/manifest",
        tags=["音声合成モデル管理"],
    )
    def get_aivm_manifest(
        aivm_uuid: Annotated[str, Path(description="音声合成モデルの UUID")]
    ) -> AivmManifest:
        """
        音声合成モデルの AIVM マニフェストを返します。
        """
        return aivm_manager.get_aivm_manifest(aivm_uuid)

    return router
