DIR=$(dirname $0)
INPUTS=$@
mkdir -p $DIR/input

for INPUT in $INPUTS; do
    echo $INPUT
    n_fields=$(awk '{print NF; exit}' $INPUT)
    if [[ $n_fields -eq 6 ]]; then
        echo "TREC format, we will need to drop the second column..."
        cat $INPUT | awk '{print $1 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' > $DIR/input/$(basename $INPUT)
    elif [[ $n_fields -eq 5 ]]; then
        echo "ARQMath-v2 format, no change."
        cp $INPUT $DIR/input/
    else
        echo "Unknown format, abort."
        exit 1
    fi
done
