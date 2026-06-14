#!/bin/bash
#
# Configure Lustre servers and clients on CloudLab using node lists.
# Reads configuration from run-in-cloudlab/config.env.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOL_DIR="$ROOT_DIR/src"
CFG_DIR="$ROOT_DIR/config"
CONFIG_FILE="${CONFIG_FILE:-$CFG_DIR/config.env}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

cloudlab_user="${SSH_USER:-}"
if [ -z "$cloudlab_user" ]; then
    echo "Error: SSH_USER is not set in config.env" >&2
    exit 1
fi

mount_path="${LUSTRE_MOUNT:-}"
if [ -z "$mount_path" ]; then
    echo "Error: LUSTRE_MOUNT is not set in config.env" >&2
    exit 1
fi
mount_path="${mount_path%/}"
fs_name="$(basename "$mount_path")"
if [ -z "$fs_name" ] || [ "$fs_name" = "/" ] || [ "$fs_name" = "." ]; then
    echo "Error: could not derive FS_NAME from LUSTRE_MOUNT=$mount_path" >&2
    exit 1
fi

# Shell snippet sourced on each remote node to find the free (non-OS) disk.
# Finds the disk backing / and picks the other sd* disk.
read -r -d '' FIND_FREE_DISK << 'SNIPPET' || true
os_disk=$(lsblk -nro PKNAME $(findmnt -n -o SOURCE /) 2>/dev/null | head -1)
free_disk=""
for d in /sys/block/sd*; do
    dname=$(basename "$d")
    if [ "$dname" != "$os_disk" ]; then
        free_disk="/dev/$dname"
        break
    fi
done
if [ -z "$free_disk" ]; then
    echo "ERROR: could not find a free disk (os_disk=$os_disk)" >&2
    exit 1
fi
echo "Detected OS disk=$os_disk, using free disk=$free_disk"
SNIPPET

servers_file="${SERVERS_FILE:-}"
if [ -z "$servers_file" ]; then
    echo "Error: SERVERS_FILE is not set in config.env" >&2
    exit 1
fi

clients_file="${CLIENTS_FILE:-}"
if [ -z "$clients_file" ]; then
    echo "Error: CLIENTS_FILE is not set in config.env" >&2
    exit 1
fi

# Resolve relative file paths from script directory for convenience.
if [[ "$servers_file" != /* ]]; then
    servers_file="$SCRIPT_DIR/$servers_file"
fi
if [[ "$clients_file" != /* ]]; then
    clients_file="$SCRIPT_DIR/$clients_file"
fi

if [ ! -f "$servers_file" ]; then
    echo "Error: servers file not found: $servers_file" >&2
    exit 1
fi
if [ ! -f "$clients_file" ]; then
    echo "Error: clients file not found: $clients_file" >&2
    exit 1
fi

# mgs node is the first line of the servers file
mgs_node="$(head -n 1 "$servers_file" | xargs)"
if [ -z "$mgs_node" ]; then
    echo "Error: first line of SERVERS_FILE is empty" >&2
    exit 1
fi

mgs_ip="$(ssh "$cloudlab_user@$mgs_node" << 'ENDSSH'
sudo su - -c '
device=$(awk "{print \$1}" /var/emulab/boot/ifmap)
ip addr show "$device" | grep -oP "inet \K[\d.]+"
'
ENDSSH
)"

if [ -z "$mgs_ip" ]; then
    echo "Error: Failed to get MGS IP" >&2
    exit 1
fi
echo "MGS IP is: $mgs_ip"

ssh -tt -p 22 "$cloudlab_user@$mgs_node" << EOF
sudo su -
$FIND_FREE_DISK

device=\$(awk '{print \$1}' /var/emulab/boot/ifmap)
existing_ip=\$(ip addr show \$device | grep -oP 'inet \K[\d.]+')

ip addr add ${mgs_ip}/24 dev \$device || true
ip link set \$device up

rmmod -f lustre || true
rmmod -f lov || true
rmmod -f mdc || true
rmmod -f lmv || true
rmmod -f ptlrpc || true
rmmod -f obdclass || true
rmmod -f ksocklnd || true
rmmod -f lnet || true
rmmod -f libcfs || true

echo "options lnet networks=tcp(\$device)" > /etc/modprobe.d/lustre.conf

modprobe libcfs
modprobe lnet
lctl network configure
lctl network up
lctl list_nids

mkfs.lustre --mgs --mdt --fsname=$fs_name --mgsnode=${mgs_ip}@tcp --index=0 --reformat \$free_disk || true

mkdir -p /mnt/mgs_mdt
mount -t lustre \$free_disk /mnt/mgs_mdt || true
exit 0
exit
EOF

server_machines=()
while IFS= read -r line || [ -n "$line" ]; do
    server_machines+=("$line")
done < "$servers_file"
start_index=0
ost_per_node=1

for machine in "${server_machines[@]}"; do
    machine="$(echo "$machine" | xargs)"
    [ -n "$machine" ] || continue
    if [ "$machine" = "$mgs_node" ]; then
        echo "Skipping OST setup on MGS node: $machine"
        continue
    fi
    echo "Running setup for $machine"

    idx1=$((start_index))
    ost1="ost${idx1}"
    ost_idx_padded="$(printf '%04d' "$idx1")"

    # Skip if this OST index already exists in the filesystem metadata.
    if ssh -T -p 22 "$cloudlab_user@$mgs_node" "sudo lctl dl 2>/dev/null | grep -q 'OST${ost_idx_padded}'"; then
        echo "Skipping $machine: OST index $idx1 (OST${ost_idx_padded}) already exists"
        start_index=$((start_index + ost_per_node))
        continue
    fi

    ssh -T -p 22 "$cloudlab_user@$machine" << EOF
sudo su -
$FIND_FREE_DISK

device=\$(awk '{print \$1}' /var/emulab/boot/ifmap)
existing_ip=\$(ip addr show \$device | grep -oP 'inet \K[\d.]+')
ip addr add \$existing_ip/24 dev \$device || true
ip link set \$device up
echo "options lnet networks=tcp(\$device)" > /etc/modprobe.d/lustre.conf
mkfs.lustre --fsname=$fs_name --ost --mgsnode=${mgs_ip}@tcp --index=$idx1 --reformat \$free_disk || true
mkdir -p /mnt/$ost1
mount -t lustre \$free_disk /mnt/$ost1 || true
if lctl list_param -N 'ost.*.ost_io.nrs_policies' >/dev/null 2>&1; then
    lctl set_param ost.*.ost_io.nrs_policies="tbf" || true
else
    echo "Skipping ost_io.nrs_policies set_param (no OST param path found)"
fi
if lctl list_param -N '*.*.job_cleanup_interval' >/dev/null 2>&1; then
    lctl set_param *.*.job_cleanup_interval=2 || true
else
    echo "Skipping job_cleanup_interval set_param (param path not found)"
fi
exit 0
exit
EOF

    start_index=$((start_index + ost_per_node))
done
echo "Done setting up servers"

while IFS= read -r machine || [ -n "$machine" ]; do
    machine="$(echo "$machine" | xargs)"
    [ -n "$machine" ] || continue
    echo "Running setup for $machine"

    ssh -T -p 22 "$cloudlab_user@$machine" << EOF
sudo su -
device=\$(awk '{print \$1}' /var/emulab/boot/ifmap)
existing_ip=\$(ip addr show \$device | grep -oP 'inet \K[\d.]+')
ip addr add \$existing_ip/24 dev \$device || true
ip link set \$device up

rmmod -f lustre || true
rmmod -f lov || true
rmmod -f mdc || true
rmmod -f lmv || true
rmmod -f ptlrpc || true
rmmod -f obdclass || true
rmmod -f ksocklnd || true
rmmod -f lnet || true
rmmod -f libcfs || true

echo "options lnet networks=tcp(\$device)" > /etc/modprobe.d/lustre.conf

modprobe libcfs
modprobe lnet
lctl network configure
lctl network up
lctl list_nids

if [ -d "/mnt/$fs_name" ]; then
    rm -rf /mnt/$fs_name
fi
mkdir -p /mnt/$fs_name/
mount -t lustre ${mgs_ip}@tcp:/$fs_name /mnt/$fs_name/ || true
exit 0
exit
EOF
done < "$clients_file"
