import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Literal


class LicenseError(Exception):
    # License違反があった場合、このエラーを出します。
    pass


class License:
    def __init__(
        self,
        package_name: str,
        package_version: str | None,
        license_name: str | None,
        license_text: str,
        license_text_type: Literal["raw", "local_address", "remote_address"],
    ):
        self.package_name = package_name
        self.package_version = package_version
        self.license_name = license_name

        if license_text_type == "raw":
            self.license_text = license_text
        elif license_text_type == "local_address":
            # ライセンステキストをローカルのライセンスファイルから抽出する
            self.license_text = Path(license_text).read_text(encoding="utf8")
        elif license_text_type == "remote_address":
            # ライセンステキストをリモートのライセンスファイルから抽出する
            with urllib.request.urlopen(license_text) as res:
                _license_text: str = res.read().decode()
                self.license_text = _license_text
        else:
            raise Exception("型で保護され実行されないはずのパスが実行されました")


def generate_licenses() -> list[License]:
    python_version = "3.11.9"
    licenses: list[License] = []
    licenses += [
        License(
            package_name="VOICEVOX ENGINE",
            package_version=None,
            license_name="LGPL license",
            license_text="https://raw.githubusercontent.com/VOICEVOX/voicevox_engine/master/LGPL_LICENSE",
            license_text_type="remote_address",
        ),
        # https://sourceforge.net/projects/open-jtalk/files/Open%20JTalk/open_jtalk-1.11/
        License(
            package_name="Open JTalk",
            package_version="1.11",
            license_name="Modified BSD license",
            license_text="docs/licenses/open_jtalk/COPYING",
            license_text_type="local_address",
        ),
        License(
            package_name="MeCab",
            package_version=None,
            license_name="Modified BSD license",
            license_text="docs/licenses/open_jtalk/mecab/COPYING",
            license_text_type="local_address",
        ),
        License(
            package_name="NAIST Japanese Dictionary",
            package_version=None,
            license_name="Modified BSD license",
            license_text="docs/licenses//open_jtalk/mecab-naist-jdic/COPYING",
            license_text_type="local_address",
        ),
        License(
            package_name="PyTorch",
            package_version="2.2.2",
            license_name="BSD-style license",
            license_text="https://raw.githubusercontent.com/pytorch/pytorch/master/LICENSE",
            license_text_type="remote_address",
        ),
        License(
            package_name="Python",
            package_version=python_version,
            license_name="Python Software Foundation License",
            license_text=f"https://raw.githubusercontent.com/python/cpython/v{python_version}/LICENSE",
            license_text_type="remote_address",
        ),
    ]

    # pip
    try:
        pip_licenses_output = subprocess.run(
            "pip-licenses "
            "--from=mixed "
            "--format=json "
            "--with-urls "
            "--with-license-file "
            "--no-license-path ",
            shell=True,
            capture_output=True,
            check=True,
            env=os.environ,
        ).stdout.decode()
    except subprocess.CalledProcessError as err:
        raise Exception(
            f"command output:\n{err.stderr and err.stderr.decode()}"
        ) from err

    licenses_json = json.loads(pip_licenses_output)
    for license_json in licenses_json:
        license = License(
            package_name=license_json["Name"],
            package_version=license_json["Version"],
            license_name=license_json["License"],
            license_text=license_json["LicenseText"],
            license_text_type="raw",
        )
        # FIXME: assert license type
        if license.license_text == "UNKNOWN":
            if license.package_name.lower() == "future":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/PythonCharmers/python-future/master/LICENSE.txt"  # noqa: B950
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "pefile":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/erocarrera/pefile/master/LICENSE"  # noqa: B950
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "pyopenjtalk-dict":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/litagin02/pyopenjtalk/master/LICENSE.md"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "python-multipart":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/andrew-d/python-multipart/master/LICENSE.txt"  # noqa: B950
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "romkan":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/soimort/python-romkan/master/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "distlib":
                with urllib.request.urlopen(
                    "https://bitbucket.org/pypa/distlib/raw/7d93712134b28401407da27382f2b6236c87623a/LICENSE.txt"  # noqa: B950
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "jsonschema":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/python-jsonschema/jsonschema/dbc398245a583cb2366795dc529ae042d10c1577/COPYING"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "lockfile":
                with urllib.request.urlopen(
                    "https://opendev.org/openstack/pylockfile/raw/tag/0.12.2/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "platformdirs":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/platformdirs/platformdirs/aa671aaa97913c7b948567f4d9c77d4f98bfa134/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "webencodings":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/gsnedders/python-webencodings/fa2cb5d75ab41e63ace691bc0825d3432ba7d694/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "antlr4-python3-runtime":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/antlr/antlr4/v4.11.1/LICENSE.txt"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "gradio_client":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/gradio-app/gradio/v3.41.0/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "jieba":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/fxsjy/jieba/v0.42.1/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "primepy":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/janaindrajit/primePy/9c98276fee5211e8761dfc03c9a1e02127e09e4a/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "pyproject_hooks":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/pypa/pyproject-hooks/v1.1.0/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "safetensors":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/huggingface/safetensors/v0.4.3/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "sentencepiece":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/google/sentencepiece/v0.2.0/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "tokenizers":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/huggingface/tokenizers/v0.19.1/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            elif license.package_name.lower() == "types-pyyaml":
                with urllib.request.urlopen(
                    "https://raw.githubusercontent.com/python/typeshed/57f3dcac8dbed008479b251512975901a0206deb/LICENSE"
                ) as res:
                    license.license_text = res.read().decode()
            else:
                # ライセンスがpypiに無い
                raise Exception(f"No License info provided for {license.package_name}")

        # soxr
        if license.package_name.lower() == "soxr":
            with urllib.request.urlopen(
                "https://raw.githubusercontent.com/dofuuz/python-soxr/v0.3.6/LICENSE.txt"
            ) as res:
                license.license_text = res.read().decode()

        licenses.append(license)

    return licenses


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()

    output_path = args.output_path

    licenses = generate_licenses()

    # dump
    out = Path(output_path).open("w") if output_path else sys.stdout
    json.dump(
        [
            {
                "name": license.package_name,
                "version": license.package_version,
                "license": license.license_name,
                "text": license.license_text,
            }
            for license in licenses
        ],
        out,
    )
