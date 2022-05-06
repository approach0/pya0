set -xe
for file_path in $@; do
    curl --progress-bar -kT $file_path -u "w32zhong:$PASS" -o /dev/null https://vault.cs.uwaterloo.ca/remote.php/dav/files/E0940394-7E80-4264-A565-AD70FBCD5378/
done
