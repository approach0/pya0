DIR=$(dirname $0)
INPUTS=$@
mkdir -p $DIR/input

if [ "$INPUTS" == "cleanup" ]; then
    set -x
    rm -f $DIR/input/*
    exit 0
fi

for INPUT in $INPUTS; do
    if [[ "$INPUT" == "swap" ]]; then
        continue
    fi

    echo $INPUT
    n_fields=$(awk '{print NF; exit}' $INPUT)
    dest_name=$(basename $INPUT)
    dest_name=$(echo $dest_name | sed -e 's/\./_/g')
    if [[ " $INPUTS " =~ " swap " ]]; then
        echo "SWAP format: Query_Id {Post_Id <=> Formula_Id} Rank Score Run"
        cat $($DIR/swap-col-2-and-3.sh $INPUT) > $DIR/input/$dest_name
    elif [[ $n_fields -eq 6 ]]; then
        echo "ARQMath task2 format: Query_Id Formula_Id Post_Id Rank Score Run"
        cat $INPUT > $DIR/input/$dest_name
    else
        echo "Unknown format, abort."
        exit 1
    fi
done
