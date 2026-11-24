#!/bin/sh
set -eu

mkdir -p dist
cp server.js run.sh dist/
cp -R public data dist/
chmod 755 dist/run.sh
