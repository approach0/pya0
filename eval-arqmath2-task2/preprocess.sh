DIR=$(dirname $0)
INPUTS=$@
mkdir -p $DIR/input

if [ "$INPUTS" == "cleanup" ]; then
    set -x
    rm -f $DIR/input/*
    exit 0
fi

for INPUT in $INPUTS; do
    echo $INPUT
    n_fields=$(awk '{print NF; exit}' $INPUT)
    dest_name=$(basename $INPUT)
    dest_name=$(echo $dest_name | sed -e 's/\./_/g')
    if [[ $n_fields -eq 6 ]]; then
        echo "ARQMath-v2 format: Query_Id Formula_Id Post_Id Rank Score Run"
        cat $INPUT > $DIR/input/$dest_name
    else
        echo "Unknown format, abort."
        exit 1
    fi
done
