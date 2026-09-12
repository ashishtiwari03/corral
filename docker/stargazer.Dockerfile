ARG CORRAL_BASE_IMAGE=corral-benchmark:latest
FROM ${CORRAL_BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/corral
COPY src /opt/corral/src
COPY pyproject.toml README.md /opt/corral/
COPY tasks/stargazer /opt/corral/tasks/stargazer
COPY tests/runtime/test_permissions.py /opt/corral/tests/runtime/test_permissions.py
RUN python -m pip install --no-cache-dir --no-deps --editable .
RUN python tasks/stargazer/scripts/install_rebound.py
RUN python -m pip install --no-cache-dir --editable tasks/stargazer pytest \
    && python -m corral.runtime.permissions --prepare-image

ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CMD ["corral", "--help"]
