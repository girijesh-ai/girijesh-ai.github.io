# Girijesh Prasad's AI Blog

Professional blog on AI, Machine Learning, and Agentic Systems.

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
bundle install

# Run local server
bundle exec jekyll serve

# Open http://localhost:4000
```

### Deploy to Git Hub Pages

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment helper
./deploy.sh
```

Your blog will be live at: `https://girijesh-ai.github.io`

## 📁 Structure

```
├── _config.yml          # Site configuration
├── Gemfile             # Ruby dependencies
├── index.md            # Homepage
├── about.md            # About page
├── _posts/             # Blog posts (YYYY-MM-DD-title.md format)
│   └── 2026-02-04-reasoning-llms-explained.md
└── assets/             # Images, CSS, etc.
    └── images/
```

## ✍️ Writing New Posts

1. Create file in `_posts/` with format: `YYYY-MM-DD-post-title.md`
2. Add front matter:

```yaml
---
layout: post
title: "Your Post Title"
date: 2026-02-04 20:00:00 +0530
categories: [AI, ML]
tags: [tag1, tag2]
author: Girijesh Prasad
excerpt: "Brief description"
---
```

3. Write content in Markdown
4. Commit and push to GitHub

## 🎨 Customization

### Change Theme Colors

Edit `_config.yml`:
```yaml
minima:
  skin: dark  # Options: auto, dark, classic, solarized
```

### Add Custom Domain

1. Create `CNAME` file with your domain
2. Update DNS settings
3. Enable HTTPS in GitHub settings

## 📊 Analytics (Optional)

Add to `_config.yml`:
```yaml
google_analytics: UA-XXXXXXXXX-X
```

## 🔗 Links

- **Live Site:** https://girijesh-ai.github.io
- **LinkedIn:** [linkedin.com/in/girijeshcse](https://linkedin.com/in/girijeshcse)
- **GitHub:** [github.com/girijesh-ai](https://github.com/girijesh-ai)

## 📝 License

Content: CC BY 4.0  
Code: MIT
