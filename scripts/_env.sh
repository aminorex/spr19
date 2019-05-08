#
# source _env.sh
#
export PYTHONPATH=$(pwd)
export PATH=$(realpath ../scripts):$PATH
# disable cuda
# export CUDA_VISIBLE_DEVICES=''
# portable location
export PROJ_ROOT=/media/sf_project/project/
# activate package set
conda activate ptbert
