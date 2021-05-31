ENV_BASE_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

prepend_path() {
    # the path is a directory and is not yet included
    if [ -d "$1" ] && [[ ":${!2}:" != *":$1:"* ]]
    then
        eval export $2="\"$1\"${!2:+\":${!2}\"}"
    fi
}

prepend_path "$ENV_BASE_DIR" PYTHONPATH

unset ENV_BASE_DIR
