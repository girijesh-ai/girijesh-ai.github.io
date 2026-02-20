# GitHub Pages Site Fixes & Notes

Reference for known issues and applied fixes for this Jekyll/GitHub Pages blog.

---

## 1. Mermaid Diagrams Not Rendering

**Date:** 2026-02-20  
**Affected Post:** `_posts/2026-02-06-llms-inferencing-explained.md`

### Root Cause
- GitHub Pages pins **Minima v2.5.1**, which has a [known bug](https://github.com/jekyll/minima/issues) where `_includes/custom-head.html` is **silently ignored**.
- Mermaid.js needs to be loaded client-side (GitHub Pages doesn't support server-side Mermaid plugins).

### Fix Applied
1. **Created `_layouts/post.html`** — custom post layout that mirrors Minima's default but appends Mermaid.js (`v11`, CDN) at the end of `<body>`.
2. **Converted `` ```mermaid `` code blocks → `<pre class="mermaid">` tags** in the post. Jekyll/kramdown renders fenced mermaid blocks as `<code>` elements that Mermaid.js can't auto-detect.
3. **HTML-escaped special chars** inside `<pre>` tags (`<br/>` → `&lt;br/&gt;`, `&` → `&amp;`) so kramdown doesn't mangle them.

### Key Files
| File | Purpose |
|------|---------|
| `_layouts/post.html` | Custom layout with Mermaid.js script |
| `_includes/custom-head.html` | **Not used** (Minima 2.5.1 bug — kept for reference) |

### How to Add Mermaid Diagrams in Future Posts
Use `<pre class="mermaid">` tags directly (not `` ```mermaid `` fences):

```html
<pre class="mermaid">
graph TD
    A[Start] --> B[End]
</pre>
```

> **Note:** Use `&lt;br/&gt;` instead of `<br/>` and `&amp;` instead of `&` inside node labels.

---

## 2. Images Not Rendering on GitHub Pages

**Date:** 2026-02-20  
**Affected Post:** `_posts/2026-02-06-llms-inferencing-explained.md`

### Root Cause
Image paths used **absolute local filesystem paths** (e.g., `/Users/girijesh/Documents/...`) which don't exist on GitHub's servers.

### Fix
- Use **site-relative paths**: `/assets/images/filename.png`
- Store all images in `assets/images/`

### Checklist for New Posts
- [ ] Images stored in `assets/images/` (or a subdirectory)
- [ ] Paths in markdown use `/assets/images/...` (not local paths)
- [ ] Post has Jekyll front matter (`layout: post`, `title`, `date`, etc.)

---

## 3. Missing Jekyll Front Matter

**Date:** 2026-02-20

### Root Cause
Post file started with `# Title` instead of YAML front matter. Jekyll requires the `---` block to process a file as a post.

### Required Front Matter Template
```yaml
---
layout: post
title: "Your Post Title"
date: 2026-MM-DD 09:00:00 +0530
categories: [AI, LLM]
tags: [tag1, tag2]
author: Girijesh Prasad
excerpt: "Brief description."
image: assets/images/thumbnail.png
---
```
