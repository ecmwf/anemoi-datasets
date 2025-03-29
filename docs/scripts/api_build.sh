

set -e

script_dir=$(dirname "${BASH_SOURCE[0]}")
docs_dir="$script_dir/.."
source_dir="$script_dir/../../src/"
cd "$source_dir"

trap 'rm -f anemoi/__init__.py' EXIT

touch anemoi/__init__.py
sphinx-apidoc -M -f -o "$docs_dir/_api" anemoi -t "$docs_dir/_templates/apidoc"
