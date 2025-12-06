#!/bin/bash
# Script to bump version and create release

set -e

CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "Current version: $CURRENT_VERSION"

if [ -z "$1" ]; then
    echo "Usage: ./bump_version.sh [major|minor|patch]"
    exit 1
fi

BUMP_TYPE=$1

IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

case $BUMP_TYPE in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo "Invalid bump type. Use: major, minor, or patch"
        exit 1
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "New version: $NEW_VERSION"

# Update pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
rm pyproject.toml.bak

# Update setup.py
sed -i.bak "s/version=\".*\"/version=\"$NEW_VERSION\"/" setup.py
rm setup.py.bak

echo "✅ Version bumped to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Commit: git add -A && git commit -m \"chore: Bump version to $NEW_VERSION\""
echo "3. Tag: git tag -a v$NEW_VERSION -m \"Release v$NEW_VERSION\""
echo "4. Push: git push && git push --tags"
echo "5. Create GitHub release to trigger PyPI publish"
