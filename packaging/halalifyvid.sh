#!/bin/sh
# HalalifyVid launcher: pass one or more videos or links, or run it without
# arguments to be asked for one. The first run installs ONNX Runtime and the
# models into the "data" folder next to this script.
here=$(cd "$(dirname "$0")" && pwd)
bin="$here/halalify"

if [ ! -e "$here/data/bin/onnxruntime/libonnxruntime.so" ]; then
    echo "First run: installing ONNX Runtime and the models. This happens once."
    echo
    "$bin" setup || { echo "Setup failed; see the messages above."; exit 1; }
    echo
fi

if [ "$#" -eq 0 ]; then
    printf "Video file or link: "
    read -r input || exit 0
    [ -n "$input" ] || exit 0
    set -- "$input"
fi

status=0
for item in "$@"; do
    "$bin" "$item" || status=1
done
exit "$status"
