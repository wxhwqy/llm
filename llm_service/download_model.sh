#!/bin/bash
# Download Qwen3.5-35B-A3B-GPTQ-Int4 from hf-mirror with resume support
MODEL_DIR="/Users/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4"
BASE_URL="https://hf-mirror.com/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/resolve/main"

mkdir -p "$MODEL_DIR"

# Expected sizes (from Content-Length headers, approximately 1.4GB each)
TOTAL_SHARDS=14

download_file() {
    local fname="$1"
    local expected_min_size="$2"  # minimum expected size in bytes

    local fpath="$MODEL_DIR/$fname"

    # Check if already complete
    if [ -f "$fpath" ]; then
        local size=$(stat -f%z "$fpath" 2>/dev/null || stat -c%s "$fpath" 2>/dev/null)
        if [ "$size" -ge "$expected_min_size" ]; then
            echo "[OK] $fname already complete ($size bytes)"
            return 0
        fi
    fi

    # Download with resume, retry until done
    local attempts=0
    while true; do
        attempts=$((attempts + 1))
        echo "[Attempt $attempts] Downloading $fname..."
        curl -L -C - --retry 3 --retry-delay 2 --connect-timeout 30 -o "$fpath" "$BASE_URL/$fname" 2>&1

        local size=$(stat -f%z "$fpath" 2>/dev/null || stat -c%s "$fpath" 2>/dev/null)
        if [ "$size" -ge "$expected_min_size" ]; then
            echo "[DONE] $fname complete ($size bytes)"
            return 0
        fi

        echo "[RETRY] $fname incomplete ($size bytes), retrying..."
        sleep 2

        if [ $attempts -ge 50 ]; then
            echo "[FAIL] $fname failed after $attempts attempts"
            return 1
        fi
    done
}

# Download all shards
for i in $(seq 1 $TOTAL_SHARDS); do
    fname=$(printf "model.safetensors-%05d-of-%05d.safetensors" $i $TOTAL_SHARDS)
    # Each shard is roughly 1.3-1.5GB, use 1GB as minimum
    download_file "$fname" 1000000000
done

# Download smaller files too
for fname in model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json; do
    if [ ! -f "$MODEL_DIR/$fname" ]; then
        echo "Downloading $fname..."
        curl -L -o "$MODEL_DIR/$fname" "$BASE_URL/$fname" 2>&1
    fi
done

echo ""
echo "=== Download Summary ==="
du -sh "$MODEL_DIR"
ls -lh "$MODEL_DIR"/model.safetensors-*.safetensors 2>/dev/null | wc -l
echo "shards downloaded"
