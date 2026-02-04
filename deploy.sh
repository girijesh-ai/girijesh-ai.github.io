#!/bin/bash

# GitHub Pages Deployment Script
# This script helps you deploy your blog to GitHub Pages

echo "🚀 GitHub Pages Blog Deployment Helper"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "_config.yml" ]; then
    echo "❌ Error: Not in blog directory (no _config.yml found)"
    echo "Please cd to girijesh-ai.github.io first"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo ""

# Step 1: Initialize git if not already
if [ ! -d ".git" ]; then
    echo "🔧 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already exists"
fi

echo ""

# Step 2: Check for remote
if ! git remote | grep -q "origin"; then
    echo "📝 No remote found. Setting up remote..."
    echo "Enter your GitHub repository URL (e.g., https://github.com/girijesh-ai/girijesh-ai.github.io.git):"
    read REPO_URL
    git remote add origin "$REPO_URL"
    echo "✅ Remote added: $REPO_URL"
else
    echo "✅ Remote already configured:"
    git remote -v
fi

echo ""

# Step 3: Add all files
echo "📦 Adding files to Git..."
git add .

# Step 4: Commit
echo ""
echo "💬 Creating commit..."
git commit -m "Initial blog setup with Reasoning LLMs post" || echo "⚠️  No changes to commit"

echo ""

# Step 5: Push
echo "🚀 Ready to push to GitHub!"
echo "What would you like to do?"
echo ""
echo "1) Push to main branch (will deploy to GitHub Pages)"
echo "2) View status only (don't push yet)"
echo "3) Cancel"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🚀 Pushing to GitHub..."
        git branch -M main
        git push -u origin main
        echo ""
        echo "✅ Pushed successfully!"
        echo ""
        echo "🎉 Your blog is deploying!"
        echo "📍 It will be live at: https://girijesh-ai.github.io"
        echo "⏱️  Wait 1-2 minutes for GitHub to build and deploy"
        echo ""
        echo "Next steps:"
        echo "1. Go to your GitHub repository"
        echo "2. Settings → Pages"
        echo "3. Verify deployment status"
        ;;
    2)
        echo "📊 Current status:"
        git status
        ;;
    3)
        echo "👋 Deployment cancelled"
        ;;
    *)
        echo "❌ Invalid choice"
        ;;
esac

echo ""
echo "📚 Documentation:"
echo "- Jekyll docs: https://jekyllrb.com/docs/"
echo "- GitHub Pages: https://docs.github.com/pages"
echo ""
echo "🛠️ Local Preview:"
echo "Run: bundle install && bundle exec jekyll serve"
echo "Then open: http://localhost:4000"
