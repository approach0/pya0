INPUT=${1-tmp.run}
tempfile=$(mktemp)

sed -e 's/^B/A/g' $INPUT > $tempfile
echo $tempfile
