
NDSU CSci 724 Spring 2019 (qua indep. study)  course project
A. Kimball and K. Davenport-Kimball

Environment:  Ubuntu 19.04, x86-64, Conda python environment

data handling scripts are in $ROOT/scripts

soft links which may be required:

ln -s ./pytorch-pretrained-BERT/data  ../data/filtered
ln -s ./pytorch-pretrained-BERT/examples/.#run_classifier.py  tony@emmy.22407:1556307143
ln -s ./supWSD/out  ../out
ln -s ./supWSD/src/main/resources  ../../resources
ln -s ./supWSD/supwsd-toolkit-1.0.0.jar  target/supwsd-toolkit-1.0.0.jar
ln -s ./supWSD/WSD_Training_Corpora  ../data/WSD_Training_Corpora
ln -s ./supWSD/WSD_Unified_Evaluation_Datasets  ../data/WSD_Unified_Evaluation_Datasets

q. v.  links.sh

scripts presently expect a clone of
  huggingface/pytorch-pretrained-BERT and/or SI3P/supWSD
in the PROJ_ROOT directory

* reproduce supWSD F scores

cd $ROOT/supWSD
. ../scripts/_setpath.sh

export WSDTRAINDATA="$PROJROOT"/data/filtered/semcor.xml
export WSDTRAINKEY="$PROJROOT"/data/filtered/semcor.key
../scripts/train.sh

export WSDTESTDATA="$PROJROOT"/data/filtered/unified.xml
export WSDTESTKEY="$PROJROOT"/data/filtered/unified.key
../scripts/test.sh

export WSDTESTDATA="$PROJROOT"/data/filtered/dev.xml
export WSDTESTKEY="$PROJROOT"/data/filtered/dev.key
../scripts/test.sh

export WSDTESTDATA="$PROJROOT"/data/filtered/test.xml
export WSDTESTKEY="$PROJROOT"/data/filtered/test.key
../scripts/test.sh

* reproduce BertForWSD F scores

cd $ROOT/pytorch-pretrained-BERT
. ../scripts/_env.sh
python -m p save=semcor.bin semcor training
for test in unified dev test; do       
    python -m p load=semcor.bin $test testing
done



