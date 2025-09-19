#!/usr/bin/env python3
import os
import json
import subprocess
import shutil
import math
from datetime import datetime

# === 配置 ===
COMFYUI_REPO_DIR = os.getenv("COMFYUI_REPO_DIR")
CUSTOM_NODES_DIR = os.getenv("CUSTOM_NODES_DIR")
GITHUB_STATS_FILE = os.path.join(os.getenv("MANAGER_REPO_DIR"), "github-stats.json")

# 修复：统一去除空格，避免匹配失败
SKIP_REPOS = {
    "https://github.com/comfyanonymous/ComfyUI",
    "https://github.com/AIDC-AI/ComfyUI-Copilot",
    "https://github.com/Comfy-Org/ComfyUI-Manager",
    "https://github.com/ltdrdata/ComfyUI-Manager",
    "https://github.com/mengsuix/comfyui-with-cust-nodes",
    "https://github.com/justUmen/Bjornulf_custom_nodes",
    "https://github.com/suix-org-1/ComfyUI",
}

DATE_SUFFIX = datetime.now().strftime("%Y%m%d")


def convert_to_ssh_url(url):
    """将 GitHub HTTPS URL 转换为 SSH URL（修复原版空格 bug）"""
    url = url.strip()
    if url.startswith("https://github.com/"):
        path = url[len("https://github.com/"):].rstrip("/")
        return f"git@github.com:{path}.git"
    elif url.startswith("http://github.com/"):
        path = url[len("http://github.com/"):].rstrip("/")
        return f"git@github.com:{path}.git"
    return url


def should_skip_repo(url):
    """判断是否应跳过该仓库（增强健壮性：strip + replace）"""
    clean_url = url.strip().rstrip("/").replace(".git", "")
    # 对 SKIP_REPOS 也做 strip 处理，避免因空格导致漏匹配
    return clean_url in {r.strip() for r in SKIP_REPOS}


def extract_repo_info(url):
    """提取 author 和 repo 名称（修复空格问题）"""
    try:
        if "github.com" in url:
            if "@" in url:
                # git@github.com:author/repo.git
                path = url.split(":")[1].replace(".git", "")
            else:
                # https://github.com/author/repo
                path = url.split("github.com/")[1].rstrip("/")
            parts = path.split("/")[:2]
            if len(parts) < 2:
                raise ValueError("路径格式错误")
            author, repo = parts[0].strip(), parts[1].strip()
            return author, repo
    except Exception:
        pass
    return "unknown", "unknown"


def calculate_decay_score(last_updated_str, half_life_days=180):
    """计算活跃度衰减因子 - 校验未来时间，直接计算"""
    if isinstance(last_updated_str, str) and last_updated_str.strip():
        # 优先处理标准格式 "2025-02-24 04:53:52"
        if ' ' in last_updated_str and len(last_updated_str) > 10:
            last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
        else:
            # 尝试其他常见格式
            formats = [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d"
            ]
            last_updated = None
            for fmt in formats:
                try:
                    last_updated = datetime.strptime(last_updated_str, fmt)
                    break
                except ValueError:
                    continue
            if last_updated is None:
                raise ValueError(f"无法解析日期格式: {last_updated_str}")

        now = datetime.now()
        if last_updated > now:
            raise ValueError(
                f"更新时间不能为未来时间: {last_updated_str} "
                f"（当前服务器时间: {now.strftime('%Y-%m-%d %H:%M:%S')}）"
            )

        days_diff = (now - last_updated).days
        decay = math.exp(-math.log(2) * days_diff / half_life_days)
        return max(decay, 0.001)  # 防止数值下溢

    else:
        raise ValueError("更新时间为空或非字符串")


def load_sorted_repos(json_path, top_n=100):
    """加载并排序 GitHub 仓库 - 基于 star 数和活跃度综合排序（严格模式：无更新时间则排除）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    repos = []
    skipped_due_to_date = 0
    skipped_due_to_skip = 0

    for url, info in data.items():
        url = url.strip()
        if should_skip_repo(url):
            skipped_due_to_skip += 1
            continue

        author, repo = extract_repo_info(url)
        stars = info.get("stars", 0)
        last_updated = info.get("last_update")  # 你的数据统一用这个字段

        # 严格模式：必须有更新时间且能解析
        try:
            if not last_updated:
                raise ValueError("更新时间字段缺失")

            decay_score = calculate_decay_score(last_updated)
            composite_score = stars * decay_score

            repos.append({
                "reference": url,
                "author": author,
                "repo": repo,
                "title": info.get("title", repo),
                "stars": stars,
                "last_updated": last_updated,
                "decay_score": decay_score,
                "composite_score": composite_score
            })
        except Exception as e:
            # 解析失败或缺失 → 直接跳过，不加入排序
            print(f"⚠️  跳过仓库（更新时间无效）: {author}/{repo} - 原因: {e}")
            skipped_due_to_date += 1
            continue

    # 按综合得分降序排序
    repos.sort(key=lambda x: x["composite_score"], reverse=True)

    # 打印统计信息
    print(f"\n📊 数据统计:")
    print(f"  总仓库数: {len(data)}")
    print(f"  跳过（在SKIP列表）: {skipped_due_to_skip}")
    print(f"  跳过（无有效更新时间）: {skipped_due_to_date}")
    print(f"  有效参与排序: {len(repos)}")

    # 打印排序信息
    print(f"\n=== 前10名仓库排序结果 ===")
    for i, repo in enumerate(repos[:10]):
        print(f"{i+1:2d}. {repo['author']}/{repo['repo']}")
        print(f"     Stars: {repo['stars']:4d} | 综合得分: {repo['composite_score']:6.2f} | "
              f"活跃度: {repo['decay_score']:.3f} | 更新: {repo['last_updated']}")
        print()

    print(f"返回前 {min(top_n, len(repos))} 个有效仓库")
    return repos[:top_n]


def clone_repo(url, author, repo):
    """克隆仓库并清理 .git（修复路径空格，不使用标记文件）"""
    # 修复：清理空格，替换为下划线，避免路径歧义
    author = author.strip().replace(" ", "_")
    repo = repo.strip().replace(" ", "_")
    folder_name = f"{author}_{repo}_{DATE_SUFFIX}"
    target_path = os.path.join(CUSTOM_NODES_DIR, folder_name)

    # 删除旧版本（不同日期）
    if os.path.exists(CUSTOM_NODES_DIR):
        for item in os.listdir(CUSTOM_NODES_DIR):
            if item.startswith(f"{author}_{repo}_") and item != folder_name:
                old_path = os.path.join(CUSTOM_NODES_DIR, item)
                try:
                    shutil.rmtree(old_path)
                    print(f"🗑️  删除旧版本: {item}")
                except Exception as e:
                    print(f"⚠️  删除旧版本失败: {item} - {e}")

    # 👇 调试日志：打印实际检查的路径
    print(f"🔍 检查路径: {target_path}")
    if os.path.exists(target_path):
        print(f"✅ 已存在目录，跳过克隆: {folder_name}")
        return True

    ssh_url = convert_to_ssh_url(url)
    print(f"⬇️  克隆: {ssh_url} -> {target_path}")

    # 设置环境变量跳过 LFS
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    result = subprocess.run(["git", "clone", ssh_url, target_path], env=env)
    if result.returncode != 0:
        print(f"❌ 克隆失败: {ssh_url}")
        # 清理可能残留的不完整目录
        if os.path.exists(target_path):
            try:
                shutil.rmtree(target_path)
                print(f"🗑️  清理克隆失败残留目录: {target_path}")
            except Exception as e:
                print(f"⚠️  清理失败: {e}")
        return False

    git_dir = os.path.join(target_path, ".git")
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir)
        print(f"🧹 删除 .git: {git_dir}")

    return True


def git_commit_and_push(message):
    """提交并推送 Git 仓库 —— 仅提交 custom_nodes 目录，强制绕过 .gitignore"""
    original_dir = os.getcwd()
    try:
        os.chdir(COMFYUI_REPO_DIR)

        # 只强制添加 custom_nodes 目录
        subprocess.run(["git", "add", "-f", CUSTOM_NODES_DIR], check=True)

        # 检查暂存区是否有变更（在 add 之后！）
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", CUSTOM_NODES_DIR],
            capture_output=True
        )
        if result.returncode == 0:
            print("📭 custom_nodes 目录无实际变更，跳过提交")
            return True

        # 有变更 → 提交
        subprocess.run(["git", "commit", "-m", message, "--quiet"], check=True)

        # 推送
        push_result = subprocess.run(
            ["git", "push"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={**os.environ, "GIT_LFS_SKIP_PUSH": "1"},
            text=True
        )
        if push_result.returncode != 0:
            print(f"💥 Git 推送失败: {push_result.stderr.strip()}")
            return False

        print("✅ Git 提交并推送成功")
        return True

    except subprocess.CalledProcessError as e:
        print(f"💥 Git 操作失败: {e}")
        return False
    finally:
        os.chdir(original_dir)


def main():
    # 确保 custom_nodes 目录存在
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)

    repos = load_sorted_repos(GITHUB_STATS_FILE)
    cloned_count = 0

    if not repos:
        print("❌ 没有找到任何有效仓库！")
        return

    print(f"\n=== 开始克隆前 {len(repos)} 个仓库 ===")
    for i, item in enumerate(repos):
        url = item["reference"]
        author = item["author"]
        repo = item["repo"]
        stars = item["stars"]
        score = item["composite_score"]

        print(f"\n[{i+1}/{len(repos)}] 处理: {author}/{repo} (Stars: {stars}, Score: {score:.2f})")

        if clone_repo(url, author, repo):
            folder_name = f"{author}_{repo}_{DATE_SUFFIX}"
            target_path = os.path.join(CUSTOM_NODES_DIR, folder_name)

            if os.path.exists(target_path) and any(os.scandir(target_path)):
                cloned_count += 1
                commit_message = f"Add custom node: {author}/{repo} ({DATE_SUFFIX})"
                if git_commit_and_push(commit_message):
                    print(f"[{cloned_count}] ✅ 成功克隆并推送: {url}")
                else:
                    print(f"[{cloned_count}] ❌ 提交推送失败: {url}")
                    # 🗑️ 关键修复：推送失败则删除克隆目录
                    try:
                        shutil.rmtree(target_path)
                        print(f"🗑️  因推送失败，已删除目录: {target_path}")
                    except Exception as e:
                        print(f"⚠️  删除失败: {e}")
            else:
                print(f"⚠️  跳过空目录或未成功克隆: {author}/{repo}")
        else:
            print(f"❌ 克隆失败: {author}/{repo}")

    print(f"\n🎉 克隆完成，共成功处理 {cloned_count} 个节点")


if __name__ == "__main__":
    main()
