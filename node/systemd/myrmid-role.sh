#!/bin/sh
# MyrmidLM: switch this PC between worker and host (run by myrmid-role.service).
role="$(cat /run/myrmid/role 2>/dev/null)"
rm -f /run/myrmid/role
case "$role" in
    worker) off=myrmid-host ;;
    host) off=myrmid-worker ;;
    *) exit 0 ;;
esac
systemctl disable --now "$off"
systemctl enable --now "myrmid-$role"
