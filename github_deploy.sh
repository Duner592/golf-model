#!/bin/bash

# Deploy web assets to gh-pages
# Run from repo root on master branch

set -e  # Exit on error

echo "Starting deployment to gh-pages..."

# Ensure we're on master and clean
if [ "$(git branch --show-current)" != "master" ]; then
    echo "Error: Must be on master branch. Run 'git checkout master'."
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working tree not clean. Commit changes first."
    exit 1
fi

# Switch to gh-pages
git checkout gh-pages

# Clear old files (safely)
echo "Clearing old files..."
git rm -rf . || true  # Ignore errors if empty

# Copy from master
echo "Copying web assets..."
git checkout master -- web/
mv web/* . 2>/dev/null || true  # Move contents to root
rmdir web 2>/dev/null || true

# Commit and push
echo "Committing and pushing..."
git add .
if [ -n "$(git status --porcelain)" ]; then
    PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "Deploy updated webpage $(date +%Y-%m-%d)"
    git push origin gh-pages
    echo "Deployment complete!"
else
    echo "No changes to deploy."
fi

# Switch back
git checkout master