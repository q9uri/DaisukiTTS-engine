# まだ Poetry で依存ライブラリがインストールされていない場合はインストールし、ライセンス一覧を生成する

set -eux

poetry install --with=license
poetry run python build_util/generate_licenses.py > resources/engine_manifest_assets/dependency_licenses.json
