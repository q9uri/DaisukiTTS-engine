# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE=ubuntu:20.04
ARG BASE_RUNTIME_IMAGE=$BASE_IMAGE

# Compile Python (version locked)
FROM ${BASE_IMAGE} AS compile-python-env

ARG DEBIAN_FRONTEND=noninteractive

RUN <<EOF
    set -eux
    apt-get update
    apt-get install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        curl \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        git
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

ARG PYTHON_VERSION=3.11.9
ARG PYENV_VERSION=v2.3.17
ARG PYENV_ROOT=/tmp/.pyenv
ARG PYBUILD_ROOT=/tmp/python-build
RUN <<EOF
    set -eux

    git clone -b "${PYENV_VERSION}" https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    PREFIX="$PYBUILD_ROOT" "$PYENV_ROOT"/plugins/python-build/install.sh
    "$PYBUILD_ROOT/bin/python-build" -v "$PYTHON_VERSION" /opt/python

    rm -rf "$PYBUILD_ROOT" "$PYENV_ROOT"
EOF

# FIXME: add /opt/python to PATH
# not working: /etc/profile read only on login shell
# not working: /etc/environment is the same
# not suitable: `ENV` is ignored by docker-compose
# RUN <<EOF
#     set -eux
#     echo "export PATH=/opt/python/bin:\$PATH" > /etc/profile.d/python-path.sh
#     echo "export LD_LIBRARY_PATH=/opt/python/lib:\$LD_LIBRARY_PATH" >> /etc/profile.d/python-path.sh
#     echo "export C_INCLUDE_PATH=/opt/python/include:\$C_INCLUDE_PATH" >> /etc/profile.d/python-path.sh
#
#     rm -f /etc/ld.so.cache
#     ldconfig
# EOF


# Runtime
FROM ${BASE_RUNTIME_IMAGE} AS runtime-env
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/aivisspeech-engine

# ca-certificates: pyopenjtalk dictionary download
# build-essential: pyopenjtalk local build
# libsndfile1: soundfile shared object for arm64
# ref: https://github.com/VOICEVOX/voicevox_engine/issues/770
RUN <<EOF
    set -eux

    apt-get update
    apt-get install -y \
        git \
        wget \
        cmake \
        ca-certificates \
        build-essential \
        gosu \
        libsndfile1
    apt-get clean
    rm -rf /var/lib/apt/lists/*

    # Create a general user
    useradd --create-home user
EOF

# Copy python env
COPY --from=compile-python-env /opt/python /opt/python

# Install Python dependencies
ADD ./requirements.txt /tmp/
RUN <<EOF
    # Install requirements
    gosu user /opt/python/bin/pip3 install -r /tmp/requirements.txt
EOF

# Add local files
ADD ./voicevox_engine /opt/aivisspeech-engine/voicevox_engine
ADD ./docs /opt/aivisspeech-engine/docs
ADD ./run.py ./presets.yaml ./default.csv ./engine_manifest.json /opt/aivisspeech-engine/
ADD ./build_util/generate_licenses.py /opt/aivisspeech-engine/build_util/
ADD ./ui_template /opt/aivisspeech-engine/ui_template
ADD ./engine_manifest_assets /opt/aivisspeech-engine/engine_manifest_assets

# Replace version
ARG AIVISSPEECH_ENGINE_VERSION=latest
RUN sed -i "s/__version__ = \"latest\"/__version__ = \"${AIVISSPEECH_ENGINE_VERSION}\"/" /opt/aivisspeech-engine/voicevox_engine/__init__.py
RUN sed -i "s/\"version\": \"999\\.999\\.999\"/\"version\": \"${AIVISSPEECH_ENGINE_VERSION}\"/" /opt/aivisspeech-engine/engine_manifest.json

# Generate licenses.json
ADD ./requirements-license.txt /tmp/
RUN <<EOF
    set -eux

    cd /opt/aivisspeech-engine

    # Define temporary env vars
    # /home/user/.local/bin is required to use the commands installed by pip
    export PATH="/home/user/.local/bin:${PATH:-}"

    gosu user /opt/python/bin/pip3 install -r /tmp/requirements-license.txt
    gosu user /opt/python/bin/python3 build_util/generate_licenses.py > /opt/aivisspeech-engine/engine_manifest_assets/dependency_licenses.json
    cp /opt/aivisspeech-engine/engine_manifest_assets/dependency_licenses.json /opt/aivisspeech-engine/licenses.json
EOF

# Keep this layer separated to use layer cache on download failed in local build
RUN <<EOF
    set -eux

    # Download openjtalk dictionary
    # try 5 times, sleep 5 seconds before retry
    for i in $(seq 5); do
        EXIT_CODE=0
        gosu user /opt/python/bin/python3 -c "import pyopenjtalk; pyopenjtalk._lazy_init()" || EXIT_CODE=$?
        if [ "$EXIT_CODE" = "0" ]; then
            break
        fi
        sleep 5
    done

    if [ "$EXIT_CODE" != "0" ]; then
        exit "$EXIT_CODE"
    fi
EOF

# Create container start shell
COPY --chmod=775 <<EOF /entrypoint.sh
#!/bin/bash
set -eux

exec "\$@"
EOF

ENTRYPOINT [ "/entrypoint.sh"  ]
CMD [ "gosu", "user", "/opt/python/bin/python3", "./run.py", "--host", "0.0.0.0" ]

# Enable use_gpu
FROM runtime-env AS runtime-nvidia-env
CMD [ "gosu", "user", "/opt/python/bin/python3", "./run.py", "--use_gpu", "--host", "0.0.0.0" ]
