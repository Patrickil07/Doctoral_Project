#!/usr/bin/env bash
# Copies one saved run folder (aggregate results, logs, Chapter 4 figures and
# tables; never microdata) to Google Drive with rclone, then checks every file
# arrived intact (size + MD5).
#
# Usage: .github/sync_drive.sh <local run folder> <run name>
#
# Needs the repository secret RCLONE_DRIVE_TOKEN: the JSON token printed by
# `rclone authorize "drive"` on your own computer (see README, "Google Drive
# copy"). Without it the sync is skipped with a notice, never an error.
# GDRIVE_FOLDER_ID is the Drive folder that receives one sub-folder per run.
set -euo pipefail

src="$1"
name="$2"
[ -d "$src" ] || { echo "::error::$src not found"; exit 1; }

if [ -z "${RCLONE_DRIVE_TOKEN:-}" ]; then
  echo "::notice::Google Drive copy skipped: repository secret RCLONE_DRIVE_TOKEN is not set"
  exit 0
fi
: "${GDRIVE_FOLDER_ID:?GDRIVE_FOLDER_ID is not set}"

command -v rclone >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq rclone; }

conf="$RUNNER_TEMP/rclone.conf"
umask 077
cat > "$conf" <<EOF
[gdrive]
type = drive
scope = drive
root_folder_id = $GDRIVE_FOLDER_ID
token = $RCLONE_DRIVE_TOKEN
EOF
trap 'rm -f "$conf"' EXIT

# The run folder holds aggregate files only; these filters are a second guard
# so that nothing resembling microdata can ever be uploaded.
filters=(--exclude "*.parquet" --exclude "*.dat" --exclude "*.dat.gz" --exclude "data/raw/**")

rclone --config "$conf" copy "$src" "gdrive:$name" --checksum "${filters[@]}" --stats-one-line -v
rclone --config "$conf" check "$src" "gdrive:$name" --one-way "${filters[@]}"
echo "copied to Google Drive: $name ($(find "$src" -type f | wc -l) files)"
