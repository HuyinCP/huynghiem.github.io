---
layout: default
title: Knowledge Graph
permalink: /knowledge_graph.html
graph: true
---

<section class="page-intro">
  <p class="eyebrow">Explore ideas</p>
  <h1>Knowledge Graph.</h1>
  <p class="lead">A map of topics and the articles connected to them. Select an article to read it.</p>
</section>

<section class="graph-section" aria-label="Interactive knowledge graph">
  <div class="graph-legend"><span><i class="legend-dot legend-topic"></i>Topic</span><span><i class="legend-dot legend-post"></i>Article</span></div>
  <svg id="knowledge-graph" role="img" aria-label="Topics connected to blog posts" viewBox="0 0 900 520" preserveAspectRatio="xMidYMid meet"></svg>
  <p class="graph-hint">Select an article node to open it. The graph grows as you add tagged posts.</p>
</section>

<script id="graph-data" type="application/json">
[{% for post in site.posts %}{"title":{{ post.title | jsonify }},"url":{{ post.url | relative_url | jsonify }},"tags":{{ post.tags | jsonify }}}{% unless forloop.last %},{% endunless %}{% endfor %}]
</script>
