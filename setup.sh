#!/bin/sh

pushd mano_pybullet
python3 -m pip install -e .
export MANO_MODELS_DIR='../mano_v1_2/models/'
python3 -m mano_pybullet.tools.gui_control
popd