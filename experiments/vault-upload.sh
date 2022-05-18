set -e
if [ -z $PASS ]; then
	echo "Error: PASS not set."
	exit 1
fi

# write password to config file so we do not leak it in 'ps aux'
cat > curlrc <<-EOF
-u "w32zhong:$PASS"
EOF

for file_path in $@; do
	curl --progress-bar -kT $file_path --config curlrc -o /dev/null https://vault.cs.uwaterloo.ca/remote.php/dav/files/E0940394-7E80-4264-A565-AD70FBCD5378/
done

rm -f curlrc
