INPUT=${1-tmp.run}
tempfile=$(mktemp)
sed -e 's/[ ]\+/\t/g' $INPUT > $tempfile
echo $tempfile
