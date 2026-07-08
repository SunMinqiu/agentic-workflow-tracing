#!/usr/bin/env bash

results_owner() {
    printf '%s' "${RESULTS_OWNER:-${SUDO_USER:-${SSH_USER:-${USER:-Minqiu}}}}"
}

default_lustre_results_root() {
    local mount_root owner
    mount_root="${MOUNT_PATH:-/mnt/lustrefs}"
    owner="$(results_owner)"
    printf '%s/%s/pi-ebpf-tracing-handoff/results' "$mount_root" "$owner"
}

require_lustre_base_out() {
    local path="$1"
    local mount_root="${MOUNT_PATH:-/mnt/lustrefs}"
    if [ -z "$path" ]; then
        echo "Error: BASE_OUT is empty" >&2
        exit 1
    fi
    if [[ "$path" != /* ]]; then
        echo "Error: BASE_OUT must be an absolute Lustre path, got: $path" >&2
        echo "       Use BASE_OUT=$(default_lustre_results_root)/<run_id>." >&2
        exit 1
    fi
    case "$path" in
        "$mount_root"/*) ;;
        *)
            echo "Error: BASE_OUT must live under $mount_root, got: $path" >&2
            echo "       Use BASE_OUT=$(default_lustre_results_root)/<run_id>." >&2
            exit 1
            ;;
    esac
    if ! awk -v m="$mount_root" '$2 == m { found = 1 } END { exit found ? 0 : 1 }' /proc/mounts; then
        echo "Error: $mount_root is not mounted. Refusing to write trace output to root/home." >&2
        exit 1
    fi
    mkdir -p "$path"
    if [ "$(df -P "$path" | awk 'NR == 2 { print $6 }')" = "/" ]; then
        echo "Error: BASE_OUT resolves to the root filesystem: $path" >&2
        echo "       Mount Lustre first, then rerun." >&2
        exit 1
    fi
}
