TSV="./latex_representation_v3"
DIR=$(dirname $0)

# Guess QREL!
if [[ $0 == *"arqmath1"* ]]; then
    QREL="topics-and-qrels/qrels.arqmath-2020-task2-visual_ids.v3.txt"
elif [[ $0 == *"arqmath2"* ]]; then
    QREL="topics-and-qrels/qrels.arqmath-2021-task2-official.v3.txt"
elif [[ $0 == *"arqmath3"* ]]; then
    QREL="topics-and-qrels/qrels.arqmath-2022-task2-official.v3.txt"
else
    echo "Failed to guess QREL file, you must specify --qrels!!!"
fi

for arg in "$@"; do
    case $arg in
    --nojudge)
      NOJUDGE=-nojudge
      shift
      ;;
    --qrels=*)
      QREL="${arg#*=}"
      shift
      ;;
    --tsv=*)
      TSV="${arg#*=}"
      shift
      ;;
    -*|--*)
      echo "Unknown option $arg"
      exit 1
      ;;
    *)
      ;;
    esac
done

mkdir -p $DIR/prime-output
wc -l $DIR/input/*

set -ex
sed -i 's/ /\t/g' $DIR/input/*

if [[ $TSV == 'skip' ]]; then
    :
else
    rm -f $DIR/prime-output/*
    if [ ! -e $TSV ]; then
        echo "TSV directory not found: $TSV"
        exit 1
    fi

    if [[ $TSV == *"v3"* ]]; then
        python $DIR/de_duplicate.py -qre $QREL -tsv $TSV -sub "$DIR/input/" -v 6 -pri "$DIR/prime-output/"
    elif [[ $TSV == *"v2"* ]]; then
        python $DIR/de_duplicate.py -qre $QREL -tsv $TSV -sub "$DIR/input/" -v 4 -pri "$DIR/prime-output/"
    else
        echo "Cannot guess the version of TSV directory: $TSV"
        exit 1
    fi
fi

python $DIR/task2_get_results.py -eva trec_eval -qre $QREL -pri "$DIR/prime-output/" -res $DIR/result.tsv $NOJUDGE
cat $DIR/result.tsv
