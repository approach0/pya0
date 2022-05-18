QREL="topics-and-qrels/qrels.arqmath-2020-task2-visual_ids.txt"
TSV="./latex_representation_v2"
DIR=$(dirname $0)

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
      TSV="${arg#*=}/"
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

if [ ! -e $DIR/visual_id_file.tsv ]; then
	gdown '1mUZ34Jx9H5LqnguGZuX2-0G_PBi4kAiS' -O $DIR/visual_id_file.tsv
fi
if [ ! -e $DIR/formulas_slt_string.tsv ]; then
	gdown '1nRrG3T2hrQY-VU0awoHSG7-g8W-6pXth' -O $DIR/formulas_slt_string.tsv
fi

python $DIR/arqmath_2020_task2_convert_runs.py -ru "$DIR/input/" -re "$DIR/prime-output/" -v $DIR/visual_id_file.tsv -q $QREL -ld $TSV -s $DIR/formulas_slt_string.tsv

python $DIR/task2_get_results.py -eva trec_eval -qre $QREL -de "$DIR/prime-output/" -res $DIR/result.tsv $NOJUDGE

cat $DIR/result.tsv | sed -e 's/[[:blank:]]/ /g'
