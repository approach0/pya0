INPUT=${1-tmp.run}
tempfile=$(mktemp)

awk 'BEGIN {OFS=" "} {print $1, $3, $2, $4, $5, $6}' $INPUT > $tempfile
echo $tempfile
