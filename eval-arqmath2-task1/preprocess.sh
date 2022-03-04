DIR=$(dirname $0)
INPUTS=$@
mkdir -p $DIR/input

if [ "$INPUTS" == "cleanup" ]; then
    set -x
    rm -f $DIR/input/*
    exit 0
fi

for INPUT in $INPUTS; do
    if [[ "$INPUT" == "filter" ]]; then
        continue
    fi

    echo $INPUT
    n_fields=$(awk '{print NF; exit}' $INPUT)
    dest_name=$(basename $INPUT)
    dest_name=$(echo $dest_name | sed -e 's/\./_/g')
    if [[ $n_fields -eq 6 ]]; then
        echo "TREC format, we will need to drop the second column..."
        cat $INPUT | awk '{print $1 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' > $DIR/input/$dest_name
    elif [[ $n_fields -eq 5 ]]; then
        echo "ARQMath-v2 format, no change."
        cp $INPUT $DIR/input/$dest_name
    else
        echo "Unknown format, abort."
        exit 1
    fi

    csv=$DIR/Task1_ARQMath2_Topic_Information.csv
    if [[ " $INPUTS " =~ " filter " ]]; then
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Dependency Text
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Dependency Formula
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Dependency Text
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Difficulty High
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Difficulty Medium
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Difficulty Low
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Category Proof
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Category Concept
        python3 $DIR/topic_filter.py $csv $DIR/input/$dest_name Category Calculation
    fi
done
