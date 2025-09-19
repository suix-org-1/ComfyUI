#!/bin/bash

# === 配置路径 ===
MANAGER_REPO_DIR="/Users/suixmeng/suix/suix-project/ComfyUI-Manager"
COMFYUI_REPO_DIR="/Users/suixmeng/suix/suix-project/suix_comfyui/ComfyUI"
CUSTOM_NODES_DIR="$COMFYUI_REPO_DIR/custom_nodes"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # 获取当前脚本所在目录

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

# === 1. 更新 ComfyUI-Manager 仓库 ===
echo "🔄 正在更新 ComfyUI-Manager 仓库..."
cd "$MANAGER_REPO_DIR" || { echo "❌ 无法进入 ComfyUI-Manager 目录"; exit 1; }
if ! git pull origin main &>/dev/null && ! git pull origin master &>/dev/null; then
    echo "❌ ComfyUI-Manager 仓库拉取失败（main 和 master 均不可用）"
    exit 1
fi
echo "✅ ComfyUI-Manager 已更新"

# === 2. 更新 ComfyUI 仓库（智能获取默认分支，静音错误）===
echo "🔄 正在更新 ComfyUI 仓库..."
cd "$COMFYUI_REPO_DIR" || { echo "❌ 无法进入 ComfyUI 目录"; exit 1; }

# 尝试自动获取远程默认分支
DEFAULT_BRANCH=$(git remote show origin 2>/dev/null | grep "HEAD branch" | awk '{print $NF}')

# 如果自动获取失败，fallback 到检测 main 或 master
if [ -z "$DEFAULT_BRANCH" ]; then
    if git ls-remote --exit-code origin main &>/dev/null; then
        DEFAULT_BRANCH="main"
    elif git ls-remote --exit-code origin master &>/dev/null; then
        DEFAULT_BRANCH="master"
    else
        echo "❌ 无法确定远程默认分支（main/master 均不存在）"
        exit 1
    fi
fi

# 拉取默认分支
if git pull origin "$DEFAULT_BRANCH" &>/dev/null; then
    echo "✅ ComfyUI 仓库已更新（分支: $DEFAULT_BRANCH）"
else
    echo "❌ git pull origin $DEFAULT_BRANCH 失败"
    exit 1
fi

# === 3. 【可选】清空 custom_nodes 目录（通过参数控制）===
if [ "$CLEAN_NODES" = true ]; then
    echo "🧹 正在清空 custom_nodes 目录..."
    if [ -d "$CUSTOM_NODES_DIR" ]; then
        # 使用 find 更安全地删除所有内容（包括隐藏文件），完全静音
        find "$CUSTOM_NODES_DIR" -mindepth 1 -delete &>/dev/null || true
        echo "✅ custom_nodes 目录已清空"
    else
        echo "❌ custom_nodes 目录不存在: $CUSTOM_NODES_DIR"
        exit 1
    fi
else
    echo "ℹ️  跳过清空 custom_nodes 目录（如需清空，请使用 --clean 参数）"
fi

# === 4. 切换回脚本目录，激活虚拟环境，运行主程序 ===
echo "🐍 切换到脚本目录并激活虚拟环境..."
cd "$SCRIPT_DIR" || { echo "❌ 无法进入脚本目录: $SCRIPT_DIR"; exit 1; }

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ 虚拟环境已激活"
else
    echo "❌ 找不到虚拟环境: venv/bin/activate"
    exit 1
fi

if [ -f "sync_comfyui_cust_nodes.py" ]; then
    echo "🚀 开始执行 sync_comfyui_cust_nodes.py ..."
    python3 sync_comfyui_cust_nodes.py
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "🎉 脚本执行成功！"
    else
        echo "❌ 脚本执行失败，退出码: $EXIT_CODE"
        exit $EXIT_CODE
    fi
else
    echo "❌ 找不到 Python 脚本: sync_comfyui_cust_nodes.py"
    exit 1
fi