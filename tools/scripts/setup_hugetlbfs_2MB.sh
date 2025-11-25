#!/usr/bin/env bash
# Setup 2-MB HugePages and mount hugetlbfs, create a 1TB file for mmap.
set -euo pipefail

#############################
# 用户可调参数
PAGES=${1:-}                         # 若留空，将按 1TB 自动计算
MNT="${MNT:-/dev/hugepages}"         # 挂载点（符合“/dev/hugepages/omni_cache”的目标路径）
HUGEPGSZ_KB=2048                     # 2MB
MAX_RETRY=10
SLEEP=2
OMNI_FILE="${OMNI_FILE:-omni_cache}" # 文件名
MAP_SIZE_BYTES="${MAP_SIZE_BYTES:-1099511627776}"  # 默认 1TB (1<<40)

#############################
# 颜色辅助
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

#############################
# 进度条/微型动画（纯 bash，尽量少依赖）
_spinner_frames='|/-\'
_spinner_idx=0
_progress_active=0

progress_bar() {
  # 用法：progress_bar <current> <total> <prefix>
  local cur=${1:-0} total=${2:-1} prefix="${3:-}"
  if (( total <= 0 )); then total=1; fi
  if (( cur > total )); then cur=$total; fi

  local cols barw percent filled empty
  cols=$(tput cols 2>/dev/null || echo 80)
  barw=$(( cols - 40 ))
  (( barw < 10 )) && barw=10

  percent=$(( cur * 100 / total ))
  filled=$(( barw * cur / total ))
  empty=$(( barw - filled ))

  # 构建条形
  local bar_fill bar_empty
  bar_fill=$(printf '%*s' "$filled" '' | tr ' ' '#')
  bar_empty=$(printf '%*s' "$empty" '' | tr ' ' ' ')

  # 旋转指示
  local spin_char=${_spinner_frames:_spinner_idx:1}
  _spinner_idx=$(( (_spinner_idx + 1) % ${#_spinner_frames} ))

  printf "\r%s [%s%s] %3d%% (%d/%d) %s" \
    "$prefix" "$bar_fill" "$bar_empty" "$percent" "$cur" "$total" "$spin_char"
  _progress_active=1
}

progress_done() {
  if (( _progress_active == 1 )); then
    printf "\n"
    _progress_active=0
  fi
}

#############################
# 工具函数
huge_sys_dir="/sys/kernel/mm/hugepages/hugepages-${HUGEPGSZ_KB}kB"
nr_path="${huge_sys_dir}/nr_hugepages"
free_path="${huge_sys_dir}/free_hugepages"
resv_path="${huge_sys_dir}/resv_hugepages"

need_pages_from_size() {
  # 向上取整，确保足额
  local size_bytes="$1"
  local sz_per_page=$((HUGEPGSZ_KB * 1024))
  echo $(( (size_bytes + sz_per_page - 1) / sz_per_page ))
}

#############################
# 1. 预留大页（带重试 + 进度条）
reserve_pages() {
  local wanted=$1
  if [[ ! -e "$nr_path" ]]; then
    log_error "This kernel does not expose ${nr_path}. Check hugepage support and page size."
    return 1
  fi

  # 首次读取当前值，用于进度条
  local actual
  actual=$(cat "$nr_path" 2>/dev/null || echo 0)

  for i in $(seq 1 $MAX_RETRY); do
    echo "$wanted" > "$nr_path"
    # 短暂等待内核调整（部分内核需要时间完成回收）
    local wait_left=$SLEEP
    while (( wait_left >= 0 )); do
      actual=$(cat "$nr_path" 2>/dev/null || echo 0)
      progress_bar "$actual" "$wanted" "Reserving 2MB HugePages (attempt $i/$MAX_RETRY)"
      # 若已经达标，立刻退出
      if [[ "$actual" -eq "$wanted" ]]; then
        progress_done
        log_info "HugePages(2MB) reserved: $actual"
        return 0
      fi
      sleep 0.3
      wait_left=$((wait_left - 1))
    done
  done
  progress_done
  actual=$(cat "$nr_path" 2>/dev/null || echo 0)
  log_error "Failed to reserve $wanted hugepages after $MAX_RETRY attempts (current=$actual)"
  return 1
}

#############################
# 2. 挂载 hugetlbfs（若已挂载则尽量复用）
mount_hugetlbfs() {
  if mountpoint -q "$MNT"; then
    # 校验是否 hugetlbfs
    if findmnt -n -o FSTYPE "$MNT" | grep -q "^hugetlbfs$"; then
      log_info "hugetlbfs already mounted at $MNT (keeping it)"
      return 0
    else
      log_info "Unmounting non-hugetlbfs at $MNT"
      umount "$MNT" || { log_error "umount failed"; return 1; }
    fi
  fi

  mkdir -p "$MNT"
  # 注意：pagesize=2M 需要内核支持多页大小的 hugetlbfs；否则该选项被忽略
  mount -t hugetlbfs \
        -o pagesize=2M,mode=0770 \
        none "$MNT"
  log_info "hugetlbfs mounted at $MNT (2 MB pages)"
}

#############################
# 3. 创建 mmap 文件（truncate 到目标大小）
create_mmap_file() {
  local file="$MNT/$OMNI_FILE"
  local size="$MAP_SIZE_BYTES"

  # 进度提示（truncate 本身很快；hugetlbfs 会基于大小进行预留，用户可根据 free/resv 观察）
  log_info "Creating hugetlbfs file: $file size=${size} bytes (~$((size/1024/1024/1024)) GB)"
  truncate -s "$size" "$file"
  chmod 660 "$file"

  # 尝试显示“预留进度”（观测 free_hugepages 下降或 resv_hugepages 上升）
  if [[ -r "$free_path" ]]; then
    local start_free end_free target_pages
    start_free=$(cat "$free_path" 2>/dev/null || echo 0)
    target_pages=$(need_pages_from_size "$MAP_SIZE_BYTES")
    # 简单观测窗口 3 秒
    local t=0
    while (( t < 10 )); do
      end_free=$(cat "$free_path" 2>/dev/null || echo 0)
      # 用“已占用 = 起始free - 当前free”作为近似进度（仅供提示，不保证严格准确）
      local used=$(( start_free - end_free ))
      (( used < 0 )) && used=0
      (( used > target_pages )) && used=$target_pages
      progress_bar "$used" "$target_pages" "Reserving pages for file"
      sleep 0.3
      t=$((t+1))
    done
    progress_done
  fi

  log_info "Created hugetlbfs file: $file size=$(stat -c%s "$file") bytes"
}

#############################
# 4. 主流程
main() {
  # 需要 root 或 sudo
  if [[ $EUID -ne 0 ]]; then
    log_error "Please run as root or with sudo"
    exit 1
  fi

  # 打印内核实际 HugePageSize
  local k_hps_kb
  k_hps_kb=$(awk '/Hugepagesize/ {print $2}' /proc/meminfo || echo 0)
  if [[ "$k_hps_kb" -ne "$HUGEPGSZ_KB" ]]; then
    log_info "Kernel Hugepagesize is ${k_hps_kb} kB; targeting 2MB pool via ${huge_sys_dir}"
  fi

  # 计算需要的页数
  local needed_pages
  if [[ -n "${PAGES:-}" ]]; then
    needed_pages="$PAGES"
    log_info "Using user-specified pages: $needed_pages (2MB each)"
  else
    needed_pages=$(need_pages_from_size "$MAP_SIZE_BYTES")
    log_info "Auto pages for ${MAP_SIZE_BYTES} bytes (~$((MAP_SIZE_BYTES/1024/1024/1024)) GB): $needed_pages (2MB each)"
  fi

  # 预留大页（带进度）
  reserve_pages "$needed_pages"

  # 打印 free/resv
  if [[ -r "$free_path" ]]; then
    log_info "free_hugepages=$(cat "$free_path"), resv_hugepages=$(cat "$resv_path" 2>/dev/null || echo N/A)"
  fi

  # 挂载
  mount_hugetlbfs

  # 创建 mmap 文件（附带进度提示）
  create_mmap_file

  log_info "HugePages setup completed successfully!"
  log_info "Verify:"
  log_info "  cat $nr_path (reserved pages)"
  log_info "  mount | grep hugetlbfs"
  log_info "  ls -lh $MNT/$OMNI_FILE"
}

# 确保异常时进度条换行不留脏行
trap 'progress_done' EXIT

main "$@"