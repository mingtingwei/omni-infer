#!/usr/bin/env bash

# Shared helpers for test runner scripts.

_timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_info() {
  echo "[INFO][$(_timestamp)] $*"
}

log_warn() {
  echo "[WARN][$(_timestamp)] $*" >&2
}
