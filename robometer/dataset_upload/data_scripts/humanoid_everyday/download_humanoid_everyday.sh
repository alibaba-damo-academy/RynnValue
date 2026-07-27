#!/usr/bin/env bash
# -------------------------------------------------------------
# download_all.sh
#
# Downloads all the Dropbox folders in the URL list below and
# extracts each one into *{DEST_ROOT}/{folder_name}*.
#
# 1️⃣ Edit the variable `DEST_ROOT` at the top of this file to
#     point wherever you want the data stored.  
#     Example: DEST_ROOT="/home/user/DropboxData"
#
# 2️⃣ Make it executable and run:
#        chmod +x download_all.sh
#        ./download_all.sh
#
# -------------------------------------------------------------

set -euo pipefail

# ------------------------------------------------------------------
# 1️⃣ Destination root – change this to wherever you want the data.
DEST_ROOT="./datasets/humanoid_everyday"   # ← modify here

# Create it if it doesn't exist (but don't overwrite an existing one)
mkdir -p "$DEST_ROOT"

# ------------------------------------------------------------------
# 2️⃣ List of URLs (one per line). Add or remove lines as needed.
URLS=(
"https://www.dropbox.com/scl/fo/r6xwxxuiwmnypzprzqza7/AA8nw-Rehsp19BgpznzX3G8/Articulated?rlkey=42llsh52wfq47r77m05mkikus&subfolder_nav_tracking=1&st=p7o1n5u8&dl=0"
"https://www.dropbox.com/scl/fo/r6xwxxuiwmnypzprzqza7/AB-bJ9d9SthpaxRq-psftIM/Basic%20manipulation?rlkey=42llsh52wfq47r77m05mkikus&e=1&st=ujqfmcrx&dl=0"
"https://www.dropbox.com/scl/fo/r6xwxxuiwmnypzprzqza7/AHhg5yc8HuZ3nUJwHEpvV7o/Deformable?rlkey=42llsh52wfq47r77m05mkikus&subfolder_nav_tracking=1&st=uuahde75&dl=0"
"https://www.dropbox.com/scl/fo/r6xwxxuiwmnypzprzqza7/AHCZsyuhfXG9BpV9OvyTUUY/Human%20robot%20interaction?rlkey=42llsh52wfq47r77m05mkikus&subfolder_nav_tracking=1&st=f0l92o3l&dl=0"
"https://www.dropbox.com/scl/fo/r6xwxxuiwmnypzprzqza7/AIaNECZ5mNu7P84jRk5IaAE/Loco-manipulation?rlkey=42llsh52wfq47r77m05mkikus&subfolder_nav_tracking=1&st=ndxg5mxe&dl=0"
"https://www.dropbox.com/scl/fo/r6xwxxuiwmnypzprzqza7/AGDzhFFpengBQRE4K_mK-_k/Precision?rlkey=42llsh52wfq47r77m05mkikus&subfolder_nav_tracking=1&st=v7wpk52e&dl=0"
"https://www.dropbox.com/scl/fo/r6xwxxuiwmnypzprzqza7/AMeSkqL7WLIuIFVzHe8T5nE/Tool%20use?rlkey=42llsh52wfq47r77m05mkikus&subfolder_nav_tracking=1&st=1dmsbxmc&dl=0"
)

# ------------------------------------------------------------------
download_and_extract() {
    local url="$1"

    # 1️⃣ Build the ZIP URL
    local zip_url="${url/&dl=0/&dl=1}"
    zip_url="${zip_url/;dl=0;/;dl=1;}"

    echo "📦 [$url] → downloading ZIP …"
    curl -L --fail "$zip_url" --output /tmp/dropbox_folder.zip

    # 2️⃣ Determine the folder name (last path segment before any query)
    local folder_name="$(basename "${url%%\?*}")"

    # 3️⃣ Full destination path: DEST_ROOT/folder_name
    local out_dir="${DEST_ROOT}/${folder_name}"
    mkdir -p "$out_dir"

    echo "📂 [$url] → extracting to '$out_dir/' …"
    unzip -o /tmp/dropbox_folder.zip -d "$out_dir" || true # ignore non-fatal warnings

    rm -f /tmp/dropbox_folder.zip
    echo "✅ [$url] done!"
}

# ------------------------------------------------------------------
for u in "${URLS[@]}"; do
    download_and_extract "$u"
done

echo "🎉 All folders downloaded and extracted into $DEST_ROOT."

