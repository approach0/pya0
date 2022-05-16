DIR=$(dirname $0)
QREL='topics-and-qrels/qrels.arqmath-2021-task2-official.txt'
TSV="./latex_representation_v2"

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
rm -f $DIR/prime-output/*

wc -l $DIR/input/*

set -ex
sed -i 's/ /\t/g' $DIR/input/*

python $DIR/de_duplicate_2021.py -qre $QREL -tsv $TSV -sub "$DIR/input/" -pri "$DIR/prime-output/"
python $DIR/task2_get_results.py -eva trec_eval -qre $QREL -pri "$DIR/prime-output/" -res $DIR/result.tsv $NOJUDGE

cat $DIR/result.tsv | sed -e 's/[[:blank:]]/ /g'
