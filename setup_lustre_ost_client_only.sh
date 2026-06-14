#!/bin/bash
#
# Recovery helper: re-runs ONLY the OST + Client steps from
# setup_lustre_simple.sh.  Use when MGS+MDT already succeeded but the
# OST or Client ssh got dropped mid-script (so you don't redo the 50min
# MDT format).
#
# Reads the same env vars as setup_lustre_simple.sh — just:
#     source ./cloudlab_env.sh
#     bash setup_lustre_ost_client_only.sh
#

set -euo pipefail

SSH_USER="${SSH_USER:-Minqiu}"
MGS_NODE="${MGS_NODE:?source cloudlab_env.sh first}"
OST_NODE="${OST_NODE:?source cloudlab_env.sh first}"
CLIENT_NODE="${CLIENT_NODE:?source cloudlab_env.sh first}"
FS_NAME="${FS_NAME:-lustrefs}"
MOUNT_PATH="${MOUNT_PATH:-/mnt/${FS_NAME}}"

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

echo "==> Getting MGS IP from $MGS_NODE"
mgs_ip="$(ssh "$SSH_USER@$MGS_NODE" << 'ENDSSH'
sudo su - -c '
device=$(awk "{print \$1}" /var/emulab/boot/ifmap)
ip addr show "$device" | grep -oP "inet \K[\d.]+"
'
ENDSSH
)"
echo "MGS IP is: $mgs_ip"

# ----------------------------------------------------------------
# OST
# ----------------------------------------------------------------
echo "==> Setting up OST on $OST_NODE"
ssh -T -p 22 "$SSH_USER@$OST_NODE" << EOF
sudo su -
$FIND_FREE_DISK

device=\$(awk '{print \$1}' /var/emulab/boot/ifmap)
existing_ip=\$(ip addr show \$device | grep -oP 'inet \K[\d.]+')
ip addr add \$existing_ip/24 dev \$device || true
ip link set \$device up
echo "options lnet networks=tcp(\$device)" > /etc/modprobe.d/lustre.conf

mkfs.lustre --fsname=$FS_NAME --ost --mgsnode=${mgs_ip}@tcp --index=0 --reformat \$free_disk || true
mkdir -p /mnt/ost0
mount -t lustre \$free_disk /mnt/ost0 || true

if lctl list_param -N 'ost.*.ost_io.nrs_policies' >/dev/null 2>&1; then
    lctl set_param ost.*.ost_io.nrs_policies="tbf" || true
fi
if lctl list_param -N '*.*.job_cleanup_interval' >/dev/null 2>&1; then
    lctl set_param *.*.job_cleanup_interval=2 || true
fi
exit 0
exit
EOF

# ----------------------------------------------------------------
# Client
# ----------------------------------------------------------------
echo "==> Setting up Client on $CLIENT_NODE"
ssh -T -p 22 "$SSH_USER@$CLIENT_NODE" << EOF
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

if [ -d "$MOUNT_PATH" ]; then
    rm -rf $MOUNT_PATH
fi
mkdir -p $MOUNT_PATH
mount -t lustre ${mgs_ip}@tcp:/$FS_NAME $MOUNT_PATH || true
exit 0
exit
EOF

echo "==> Done. Client should now have $FS_NAME mounted at $MOUNT_PATH on $CLIENT_NODE"
