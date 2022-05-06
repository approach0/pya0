for file_path in $@; do
	awk 'BEGIN {OFS="\t"} {print $1, $3, $4, $5, $6}' $file_path > $file_path.drop-col1.run
done
