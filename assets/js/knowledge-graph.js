(() => {
  const svg = document.getElementById("knowledge-graph");
  const source = document.getElementById("graph-data");
  if (!svg || !source) return;

  let posts;
  try { posts = JSON.parse(source.textContent); } catch { return; }

  const ns = "http://www.w3.org/2000/svg";
  const width = 900;
  const height = 520;
  const center = { x: width / 2, y: height / 2 };
  const tags = [...new Set(posts.flatMap(post => post.tags || []))];
  const elements = (name, attributes = {}) => {
    const element = document.createElementNS(ns, name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    return element;
  };
  const position = (index, count, radius, offset = -Math.PI / 2) => ({
    x: center.x + Math.cos(offset + (2 * Math.PI * index) / Math.max(count, 1)) * radius,
    y: center.y + Math.sin(offset + (2 * Math.PI * index) / Math.max(count, 1)) * radius
  });
  const tagPositions = new Map(tags.map((tag, index) => [tag, position(index, tags.length, 135)]));
  const postPositions = posts.map((_, index) => position(index, posts.length, 215, -Math.PI / 2 + Math.PI / Math.max(posts.length, 1)));

  const lines = elements("g", { class: "graph-lines" });
  const nodes = elements("g", { class: "graph-nodes" });
  svg.append(lines, nodes);

  for (const point of tagPositions.values()) lines.append(elements("line", { x1: center.x, y1: center.y, x2: point.x, y2: point.y }));
  posts.forEach((post, index) => {
    for (const tag of post.tags || []) {
      const point = tagPositions.get(tag);
      if (point) lines.append(elements("line", { x1: point.x, y1: point.y, x2: postPositions[index].x, y2: postPositions[index].y }));
    }
  });

  const addNode = (point, label, kind, href) => {
    const group = elements(href ? "a" : "g", { class: `graph-node graph-${kind}` });
    if (href) group.setAttribute("href", href);
    group.append(elements("circle", { cx: point.x, cy: point.y, r: kind === "center" ? 39 : kind === "topic" ? 27 : 23 }));
    const text = elements("text", { x: point.x, y: point.y + (kind === "center" ? 62 : 46), "text-anchor": "middle" });
    text.textContent = label.length > 30 ? `${label.slice(0, 27)}…` : label;
    group.append(text);
    if (href) {
      const title = elements("title");
      title.textContent = label;
      group.append(title);
    }
    nodes.append(group);
  };

  addNode(center, "Knowledge", "center");
  for (const [tag, point] of tagPositions) addNode(point, tag, "topic");
  posts.forEach((post, index) => addNode(postPositions[index], post.title, "post", post.url));

  if (!posts.length) {
    const message = elements("text", { x: center.x, y: center.y + 86, "text-anchor": "middle", class: "graph-empty" });
    message.textContent = "Add a tagged blog post to grow this graph.";
    svg.append(message);
  }
})();
