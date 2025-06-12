

set -e

script_dir=$(dirname "${BASH_SOURCE[0]}")
docs_dir="$script_dir/.."
source_dir="$script_dir/../../src/"


trap 'rm -f $source_dir/anemoi/__init__.py' EXIT

touch "$source_dir/anemoi/__init__.py"
sphinx-apidoc -M -f -o "$docs_dir/_api" "$source_dir/anemoi" -t "$docs_dir/_templates/apidoc"
