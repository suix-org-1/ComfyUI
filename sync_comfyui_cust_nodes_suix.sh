#!/bin/bash

# === 获取脚本所在目录 ===
export COMFYUI_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUSTOM_NODES_DIR="$COMFYUI_REPO_DIR/custom_nodes"

# === 参数解析 ===
CLEAN_NODES=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_NODES=true
            shift
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo "用法: $0 [--clean|-c]  # --clean 表示清空 custom_nodes 目录"
            exit 1
            ;;
    esac
done

# === 清空 custom_nodes 目录（通过参数控制）===
if [ "$CLEAN_NODES" = true ]; then
    echo "🧹 正在清空 custom_nodes 目录..."
    if [ -d "$CUSTOM_NODES_DIR" ]; then
        # 使用 find 更安全地删除所有内容（包括隐藏文件），完全静音
        find "$CUSTOM_NODES_DIR" -mindepth 1 -maxdepth 1 -delete &>/dev/null || true
        echo "✅ custom_nodes 目录已清空"
    else
        echo "❌ custom_nodes 目录不存在: $CUSTOM_NODES_DIR"
        exit 1
    fi
else
    echo "ℹ️  跳过清空 custom_nodes 目录（如需清空，请使用 --clean 参数）"
fi

# === 运行主程序 ===
if [ -f "sync_comfyui_cust_nodes_suix.py" ]; then
    echo "🚀 开始执行 sync_comfyui_cust_nodes_suix.py ..."
    python3 sync_comfyui_cust_nodes_suix.py
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "🎉 脚本执行成功！"
    else
        echo "❌ 脚本执行失败，退出码: $EXIT_CODE"
        exit $EXIT_CODE
    fi
else
    echo "❌ 找不到 Python 脚本: sync_comfyui_cust_nodes_suix.py"
    exit 1
fi
