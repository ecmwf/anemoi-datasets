
script_dir=$(dirname "${BASH_SOURCE[0]}")
docs_dir="$script_dir/.."
source_dir="$script_dir/../../src/"

sphinx-apidoc -M -f -o "$docs_dir/_api" "$source_dir/anemoi" -t "$docs_dir/_templates/apidoc"
