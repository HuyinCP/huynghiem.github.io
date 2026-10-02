---
layout: default
title: Blog
permalink: /blogs.html
---

<section class="page-intro">
  <p class="eyebrow">Technical blog</p>
  <h1>Things I'm exploring.</h1>
  <p class="lead">Articles, tutorials, and longer notes from what I'm learning and building.</p>
</section>

<section aria-label="Blog posts">
  {% for post in site.posts %}
    <a class="entry" href="{{ post.url | relative_url }}">
      <span class="entry-main"><strong>{{ post.title | escape }}</strong><span>{{ post.description | default: post.excerpt | strip_html | truncate: 180 }}</span></span>
      <time datetime="{{ post.date | date: '%Y-%m-%d' }}">{{ post.date | date: '%b %-d, %Y' }}</time>
    </a>
  {% endfor %}
</section>
