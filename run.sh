#!/bin/sh
# AIHawk from this folder, on Python 3.14.
exec uvx --python 3.14 --refresh-package aihawk --from "$(cd "$(dirname "$0")" && pwd)" aihawk "$@"
