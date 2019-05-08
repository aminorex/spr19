
# source this file into shell environment to set variables in environment

export PROJROOT="/media/sf_project/project"

export CLASSPATH="$PROJROOT/supWSD/target/supwsd-toolkit-1.0.0.jar"
for f in	"$PROJROOT"/supWSD/target/dependency-jars/*.jar; do
    export CLASSPATH="${CLASSPATH}:$f"
done

export WSDTRAINDATA="$PROJROOT"/data/WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml
export WSDTRAINKEY="$PROJROOT"/data/WSD_Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt

export WSDTESTDATA="$PROJROOT"/data/WSD_Unified_Evaluation_Datasets/ALL/ALL.data.xml
export WSDTESTKEY="$PROJROOT"/data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt

