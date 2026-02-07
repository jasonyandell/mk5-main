# Modal Volumes

Distributed file system for persistent storage across function runs.

## Creating Volumes

```bash
# CLI
modal volume create my-volume

# Volumes v2 (beta - more files, higher throughput)
modal volume create my-volume --version=2
```

```python
# In code
volume = modal.Volume.from_name("my-volume", create_if_missing=True)
```

## Mounting Volumes

```python
@app.function(volumes={"/data": volume})
def process():
    # Write to mounted path
    Path("/data/result.txt").write_text("hello")

    # IMPORTANT: writing to /result.txt saves LOCALLY, not to volume!
    # Always use the mount path prefix
```

## Commits and Reloads

### Commits

Commits persist changes to durable storage:

```python
# Explicit commit
volume.commit()

# Auto-commit (recommended)
# - Background commits every few seconds
# - Final commit on container shutdown
# - No explicit commit needed for most use cases
```

**Performance warning**: Per-write commits add ~1 second overhead each!

```python
# BAD - 100 writes = 100 seconds of commit overhead
for i in range(100):
    write_file(f"/data/file_{i}.txt", data)
    volume.commit()

# GOOD - let Modal batch commits
for i in range(100):
    write_file(f"/data/file_{i}.txt", data)
# Auto-commits on shutdown
```

### Reloads

Fetch latest volume state (after another container modified it):

```python
@app.function(volumes={"/data": volume})
def read_latest():
    volume.reload()  # Fetch latest changes
    return Path("/data/latest.txt").read_text()
```

**Constraint**: Cannot reload while files are open:
```python
# This will fail with "volume busy"
with open("/data/file.txt") as f:
    volume.reload()  # ERROR!
```

## Concurrent Access

### V1 Volumes
- Max ~5 concurrent writers recommended
- Last-write-wins for same-file conflicts
- Performance degrades with many files (>50,000)

### V2 Volumes (Beta)
- Hundreds of concurrent writers (distinct files)
- Better random access patterns
- No file count limits (but max 32,768 per directory)
- Still last-write-wins for same file

### Safe Concurrent Pattern

Write distinct files per container:

```python
@app.function(volumes={"/data": volume})
def process(task_id: int):
    # Each container writes unique file - no conflicts
    output_path = Path(f"/data/task_{task_id:08d}.parquet")
    write_result(output_path, result)
```

### Unsafe Pattern

```python
# DON'T - multiple containers writing same file
@app.function(volumes={"/data": volume})
def process(task_id: int):
    # Race condition! Last writer wins, others lost
    with open("/data/shared_log.txt", "a") as f:
        f.write(f"Task {task_id} done\n")
```

## CLI Operations

```bash
# List volume contents
modal volume ls my-volume

# Download file
modal volume get my-volume remote/path.txt local/path.txt

# Upload file
modal volume put my-volume local/file.txt remote/file.txt

# Delete volume (WARNING: irreversible!)
modal volume delete my-volume
```

## Download Pattern

```python
@app.function(volumes={"/data": volume})
def download_all(output_dir: str):
    import shutil
    from pathlib import Path

    volume.reload()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for f in Path("/data").glob("**/*.parquet"):
        rel = f.relative_to("/data")
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest)
        print(f"Downloaded: {rel}")

@app.local_entrypoint()
def download(output_dir: str = "./data"):
    download_all.remote(output_dir)
```

## Batch Upload

```python
# Efficient multi-file upload
with volume.batch_upload() as batch:
    batch.put_file("local1.txt", "/remote/file1.txt")
    batch.put_file("local2.txt", "/remote/file2.txt")
    batch.put_directory("./local_dir/", "/remote/dir/")
```

## Performance Tips

1. **Avoid per-write commits** - Let Modal auto-commit
2. **Write distinct files** - Avoid concurrent writes to same file
3. **Use V2 for many files** - V1 degrades past 50,000 files
4. **Batch uploads** - Use `batch_upload()` for multiple files
5. **Reload before reading** - If another container may have written

## Volume Limits

| Metric | V1 | V2 |
|--------|----|----|
| Max files | ~500,000 | Unlimited* |
| Max file size | - | 1 TiB |
| Files per directory | - | 32,768 |
| Concurrent writers | ~5 | Hundreds |
| Bandwidth | 2.5 GB/s | 2.5 GB/s |

*No total limit, but directories capped at 32,768 files
