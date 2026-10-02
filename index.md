---
layout: default
title: Home
permalink: /
---

<section class="hero">
  <p class="eyebrow">Hello 👋</p>
  <h1>I'm Huy.</h1>
  <p class="lead">A little corner of the internet where I share what I'm learning, the projects I'm building, and notes I want to keep.</p>
  <div class="hero-actions">
    <a class="button" href="{{ '/blogs.html' | relative_url }}">Read the blog <span aria-hidden="true">↗</span></a>
    <a class="text-link" href="{{ '/projects.html' | relative_url }}">Explore projects <span aria-hidden="true">→</span></a>
  </div>
</section>

<section class="home-section" aria-labelledby="recent-posts">
  <div class="section-heading"><h2 id="recent-posts">Recent writing</h2><a href="{{ '/blogs.html' | relative_url }}">View all →</a></div>
  {% for post in site.posts limit:3 %}
    <a class="entry" href="{{ post.url | relative_url }}">
      <span class="entry-main"><strong>{{ post.title | escape }}</strong><span>{{ post.description | default: post.excerpt | strip_html | truncate: 130 }}</span></span>
      <time datetime="{{ post.date | date: '%Y-%m-%d' }}">{{ post.date | date: '%b %-d, %Y' }}</time>
    </a>
  {% endfor %}
</section>
