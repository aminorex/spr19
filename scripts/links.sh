#! /usr/bin/env bash

test ! -f $PROJ_ROOT/data/filtered && ln -s ./pytorch-pretrained-BERT/data  $PROJ_ROOT/data/filtered
test ! -f $PROJ_ROOT/out && ln -s ./supWSD/out  $PROJ_ROOT/out
test ! -f $PROJ_ROOT/supWSD/resources && ln -s $PROJ_ROOT/supWSD/src/main/resources  $PROJ_ROOT/supWSD/resources
test ! -f $PROJ_ROOT/target/supwsd-toolkit-1.0.0.jar && ln -s $PROJ_ROOT/supWSD/supwsd-toolkit-1.0.0.jar  $PROJ_ROOT/target/supwsd-toolkit-1.0.0.jar

