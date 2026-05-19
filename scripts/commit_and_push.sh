#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/commit_and_push.sh [commit message]
  scripts/commit_and_push.sh -m "commit message"
  scripts/commit_and_push.sh --no-push -m "commit message"

Safely commits current edits on master, rebases over workflow commits from
origin/master, then pushes. Obvious local-only files such as .env and Office
lock files are left unstaged.
EOF
}

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-master}"
COMMIT_MSG=""
NO_PUSH=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -m|--message)
      shift
      if [[ "$#" -eq 0 ]]; then
        echo "Missing commit message after -m/--message." >&2
        exit 2
      fi
      COMMIT_MSG="$1"
      ;;
    --no-push)
      NO_PUSH=true
      ;;
    --)
      shift
      COMMIT_MSG="$*"
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      COMMIT_MSG="$*"
      break
      ;;
  esac
  shift
done

COMMIT_MSG="${COMMIT_MSG:-Update project files}"

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Not inside a git repository." >&2
  exit 1
}
cd "$repo_root"

die() {
  echo "Error: $*" >&2
  exit 1
}

ensure_no_git_operation() {
  local git_dir path
  git_dir="$(git rev-parse --git-dir)"
  for path in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG; do
    if [[ -e "$git_dir/$path" ]]; then
      die "Git operation in progress ($path). Finish or abort it before running this helper."
    fi
  done
}

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$BRANCH" ]]; then
  die "Current branch is '${current_branch:-detached HEAD}', expected '$BRANCH'. Run: git checkout $BRANCH"
fi

ensure_no_git_operation

echo "Syncing $BRANCH with $REMOTE/$BRANCH before staging..."
git pull --rebase --autostash "$REMOTE" "$BRANCH"

add_paths=(
  "."
  ":(exclude).env"
  ":(exclude).env.*"
  ":(exclude)**/.env"
  ":(exclude)**/.env.*"
  ":(exclude)~$*"
  ":(exclude)**/~$*"
)

git add -A -- "${add_paths[@]}"

protected_files="$(git diff --cached --name-only -- | grep -E '(^|/)\.env($|\.)|(^|/)~\$' || true)"
if [[ -n "$protected_files" ]]; then
  echo "Leaving protected local-only files unstaged:"
  while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    echo "  $file"
    git restore --staged -- "$file"
  done <<< "$protected_files"
fi

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  echo "Committing changes: $COMMIT_MSG"
  git commit -m "$COMMIT_MSG"
fi

if [[ "$NO_PUSH" == "true" ]]; then
  echo "Committed locally; --no-push was set."
  exit 0
fi

for attempt in 1 2 3; do
  echo "Final sync before push (attempt $attempt)..."
  git pull --rebase --autostash "$REMOTE" "$BRANCH"

  echo "Pushing $BRANCH to $REMOTE..."
  if git push "$REMOTE" "$BRANCH"; then
    git fetch "$REMOTE" "$BRANCH" >/dev/null 2>&1 || true
    echo "Done. $BRANCH is pushed."
    exit 0
  fi

  if [[ "$attempt" -lt 3 ]]; then
    echo "Push failed, likely because $REMOTE/$BRANCH moved. Rebasing and retrying..."
    sleep 2
  fi
done

die "Push failed after 3 attempts. Run 'git status --short --branch' and inspect the error above."
