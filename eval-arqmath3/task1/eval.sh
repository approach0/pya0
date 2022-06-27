DIR=$(dirname $0)
QREL='topics-and-qrels/qrels.arqmath-2022-task1-official.txt'

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
mkdir -p $DIR/trec-output
rm -f $DIR/trec-output/*

wc -l $DIR/input/*

set -ex
sed -i 's/ /\t/g' $DIR/input/*

python3 $DIR/arqmath_to_prim_task1.py -qre $QREL  -sub "$DIR/input/" -tre $DIR/trec-output/ -pri $DIR/prime-output/
python3 $DIR/task1_get_results.py -eva "trec_eval" -qre $QREL -pri $DIR/prime-output/ -res "$DIR/result.tsv" $NOJUDGE

cat $DIR/result.tsv | sed -e 's/[[:blank:]]/ /g'