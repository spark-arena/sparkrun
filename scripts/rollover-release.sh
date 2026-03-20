#!/bin/bash
set -e

# rollover-release.sh - Automates main-develop branch workflow for oss-sparkrun
# 
# process:
#   - git fetch
#   - update main to match origin/main
#   - rebase develop onto origin/main
#   - tag release with vXXX matching current version in versions.yaml
#   - push tag to origin

# Script location and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Version source
VERSIONS_FILE="$REPO_ROOT/versions.yaml"

cd "$REPO_ROOT"

# Check for dirty worktree
if ! git diff-index --quiet HEAD --; then
    echo "ERROR: Your worktree is dirty. Please commit or stash your changes before running this script."
    exit 1
fi

# 1. git fetch
echo "Step 1: Fetching from origin..."
git fetch origin

# 2. Update main to match origin/main
echo "Step 2: Updating local 'main' branch to match 'origin/main'..."
# Check if main exists locally
if git show-ref --verify --quiet refs/heads/main; then
    git checkout main
    git reset --hard origin/main
else
    echo "Local 'main' branch does not exist. Creating it from origin/main..."
    git branch main origin/main
fi

# 3. Rebase develop onto origin/main
echo "Step 3: Rebasing 'develop' onto 'origin/main'..."
git checkout develop
git rebase origin/main

# 4. Extract version and tag
if [ ! -f "$VERSIONS_FILE" ]; then
    echo "ERROR: $VERSIONS_FILE not found."
    exit 1
fi

# Extract version for 'sparkrun'
VERSION=$(grep "^sparkrun:" "$VERSIONS_FILE" | cut -d: -f2 | xargs)
TAG="v$VERSION"

if [ -z "$VERSION" ]; then
    echo "ERROR: Could not extract version from $VERSIONS_FILE."
    exit 1
fi

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "NOTICE: Tag $TAG already exists."
    read -p "Do you want to RE-APPLY and push tag $TAG to origin? (y/N) " -n 1 -r
    FORCE_TAG="-f"
else
    echo "NOTICE: Tag $TAG does not exist."
    read -p "Do you want to APPLY and push tag $TAG to origin? (y/N) " -n 1 -r
    FORCE_TAG=""
fi
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting tag application and push. Your local repo is updated (rebased)."
    exit 0
fi

echo "Step 4: Tagging with $TAG..."
git tag $FORCE_TAG -a "$TAG" -m "Release $TAG"

# 5. Push tag to origin
echo "Step 5: Pushing tag $TAG to origin..."
git push origin "$TAG" $FORCE_TAG

echo "Done! Workflow completed successfully."
echo "Current branch: develop (rebased onto origin/main)"
echo "Tag $TAG pushed to origin."
