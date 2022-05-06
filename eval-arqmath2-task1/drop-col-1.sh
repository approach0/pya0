INPUT=${1-tmp.run}
tempfile=$(mktemp)
awk 'BEGIN {OFS="\t"} {print $1, $3, $4, $5, $6}' $INPUT > $tempfile
echo $tempfile
