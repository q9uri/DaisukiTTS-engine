import logging
import os
import platform
from typing import Literal

import GPUtil
import psutil
from cpuinfo import get_cpu_info

from voicevox_engine import __version__
from voicevox_engine.logging import logger

# 生成済みのユーザーエージェント文字列をキャッシュする
__user_agent_cache: str | None = None


def generate_user_agent(inference_type: Literal["CPU", "GPU"] = "CPU") -> str:
    """
    ユーザーエージェント文字列を生成する。
    エラーが発生した場合でも、最低限の情報を含むユーザーエージェント文字列を返す。
    """
    global __user_agent_cache

    if __user_agent_cache is not None:
        return __user_agent_cache

    def get_os_version() -> str:
        """
        OS バージョンを取得する。
        エラー時は 'Unknown' を返す。
        """
        try:
            os_name = platform.system()
            if os_name == "Windows":
                try:
                    wv = platform.win32_ver()
                    return f"Windows/{wv[1]}"
                except Exception as e:
                    logger.warning("Failed to get Windows version: %s", e)
                    return "Windows/Unknown"
            elif os_name == "Darwin":
                try:
                    ver = platform.mac_ver()[0]
                    return f"macOS/{ver}" if ver else "macOS/Unknown"
                except Exception as e:
                    logger.warning("Failed to get macOS version: %s", e)
                    return "macOS/Unknown"
            elif os_name == "Linux":
                try:
                    kernel = platform.release()
                    try:
                        with open("/etc/os-release") as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.startswith("PRETTY_NAME="):
                                    distro = line.split("=")[1].strip().strip('"')
                                    return f"Linux/{distro} (Kernel: {kernel})"
                    except Exception as e:
                        logger.warning("Failed to read /etc/os-release: %s", e)
                        return f"Linux/{kernel}"
                except Exception as e:
                    logger.warning("Failed to get Linux kernel version: %s", e)
                    return "Linux/Unknown"
            return f"{os_name}/Unknown"
        except Exception as e:
            logger.error("Failed to get OS information: %s", e)
            return "OS/Unknown"

    def get_architecture() -> str:
        """
        アーキテクチャ情報を取得する。
        エラー時は 'Unknown' を返す。
        """
        try:
            return platform.machine()
        except Exception as e:
            logger.error("Failed to get architecture information: %s", e)
            return "Unknown"

    def get_cpu_name() -> str:
        """
        CPU 名を取得する。
        エラー時は 'Unknown' を返す。
        """
        try:
            cpu_info = get_cpu_info()
            return cpu_info.get("brand_raw", "Unknown")
        except Exception as e:
            logger.error("Failed to get CPU information: %s", e)
            return "Unknown"

    def get_gpu_names() -> list[str]:
        """
        GPU 名を取得する。
        複数の GPU がある場合、すべての名前をリストで返す。
        エラー時は ['Unknown'] を返す。
        """
        try:
            os_name = platform.system()
            if os_name == "Windows":
                try:
                    import wmi  # type: ignore

                    w = wmi.WMI()
                    gpus = w.Win32_VideoController()
                    names = [gpu.Name for gpu in gpus if hasattr(gpu, "Name")]
                    return names if names else ["Unknown"]
                except Exception as e:
                    logger.warning("Failed to get Windows GPU information: %s", e)
                    return ["Unknown"]
            elif os_name == "Linux":
                try:
                    gpus = GPUtil.getGPUs()
                    names = [gpu.name for gpu in gpus if hasattr(gpu, "name")]
                    return names if names else ["NoGPU"]
                except Exception as e:
                    logger.warning("Failed to get Linux GPU information: %s", e)
                    return ["Unknown"]
            return ["Unknown"]
        except Exception as e:
            logger.error("Failed to get GPU information: %s", e)
            return ["Unknown"]

    def get_memory_info() -> tuple[float | None, float | None]:
        """
        メモリ情報 (全体のメモリ量と使用可能なメモリ量) を取得する。
        エラー時は None, None を返す。
        """
        try:
            vm = psutil.virtual_memory()
            total_gb = round(vm.total / (1024**3), 1)
            available_gb = round(vm.available / (1024**3), 1)
            return total_gb, available_gb
        except Exception as e:
            logger.error("Failed to get memory information: %s", e)
            return None, None

    def is_docker() -> bool:
        """
        Docker コンテナ内で実行されているかを判定する。
        エラー時は False を返す。
        """
        try:
            if os.path.exists("/.dockerenv"):
                return True
            try:
                with open("/proc/1/cgroup", "r") as f:
                    for line in f:
                        if "docker" in line or "kubepods" in line:
                            return True
            except (FileNotFoundError, PermissionError) as e:
                logger.debug("Docker check - could not read cgroup file: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to check Docker environment: %s", e)
            return False

    try:
        # OS・アーキテクチャ・CPU 名
        os_version = get_os_version()
        arch = get_architecture()
        cpu_name = get_cpu_name()

        # GPU 情報の取得
        # Mac では GPU 情報が取れないので、CPU 名を GPU 名として扱う
        gpu_names = get_gpu_names()
        gpu_info = ", ".join(gpu_names)
        if gpu_info == "Unknown" and platform.system() == "Darwin":
            gpu_info = cpu_name

        # メモリ情報の取得
        total_gb, available_gb = get_memory_info()
        mem_info = (
            f"{available_gb}GB:{total_gb}GB"
            if total_gb is not None and available_gb is not None
            else "Unknown"
        )

        # Docker 判定
        docker_flag = " Docker;" if is_docker() else ""

        # ユーザーエージェント文字列を生成
        user_agent = (
            f"AivisSpeech-Engine/{__version__} "
            f"({os_version}; {arch};"
            f"{docker_flag} "
            f"CPU/{cpu_name}; "
            f"GPU/{gpu_info}; "
            f"Memory/{mem_info}; "
            f"Inference/{inference_type})"
        )

        __user_agent_cache = user_agent
        return user_agent

    except Exception as e:
        # 最悪の場合のフォールバック
        logger.error("Failed to generate user agent string: %s", e)
        generic_user_agent = f"AivisSpeech-Engine/{__version__}"
        __user_agent_cache = generic_user_agent
        return generic_user_agent


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(level=logging.DEBUG)
    print(generate_user_agent("CPU"))
