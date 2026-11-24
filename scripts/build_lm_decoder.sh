#!/bin/bash
set -ex
cd /mnt/20TB/asaccoia/brain2text
[ -d .venv-lm ] || ~/.pyenv/versions/3.10.16/bin/python -m venv .venv-lm
.venv-lm/bin/pip install -q --upgrade pip "setuptools<70" wheel numpy==1.26.4 scipy jiwer tqdm
cd repos/speechBCI/LanguageModelDecoder/runtime/server/x86
export MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32
/mnt/20TB/asaccoia/brain2text/.venv-lm/bin/python setup.py install
cd /mnt/20TB/asaccoia/brain2text && .venv-lm/bin/python -c "import lm_decoder; print('lm_decoder OK', lm_decoder.__file__)"
