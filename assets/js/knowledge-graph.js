(() => {
  const svg = document.getElementById("knowledge-graph");
  const source = document.getElementById("graph-data");
  if (!svg || !source) return;

  let data;
  try {
    data = JSON.parse(source.textContent);
  } catch (error) {
    console.error("Unable to read knowledge graph data.", error);
    return;
  }

  const namespace = "http://www.w3.org/2000/svg";
  const nodesById = new Map(data.nodes.map((node) => [node.id, node]));
  const nodeElements = new Map();
  const linkElements = [];
  const cards = [...document.querySelectorAll(".graph-project-card")];
  const projectCount = data.nodes.filter((node) => node.type === "project").length;
  const topicCount = data.nodes.filter((node) => node.type === "topic").length;
  document.getElementById("graph-project-count").textContent = `${projectCount} ${projectCount === 1 ? "project" : "projects"}`;
  document.getElementById("graph-topic-count").textContent = `${topicCount} ${topicCount === 1 ? "topic" : "topics"}`;

  const createSvgElement = (name, attributes = {}) => {
    const element = document.createElementNS(namespace, name);
    for (const [key, value] of Object.entries(attributes)) {
      element.setAttribute(key, value);
    }
    return element;
  };

  const linksLayer = createSvgElement("g", { class: "graph-links" });
  const nodesLayer = createSvgElement("g", { class: "graph-map-nodes" });
  svg.append(linksLayer, nodesLayer);

  for (const link of data.links) {
    const sourceNode = nodesById.get(link.source);
    const targetNode = nodesById.get(link.target);
    if (!sourceNode || !targetNode) continue;

    const midpointX = (sourceNode.x + targetNode.x) / 2;
    const midpointY = (sourceNode.y + targetNode.y) / 2 - 18;
    const path = createSvgElement("path", {
      d: `M ${sourceNode.x} ${sourceNode.y} Q ${midpointX} ${midpointY} ${targetNode.x} ${targetNode.y}`,
      class: "graph-link"
    });
    linksLayer.append(path);
    linkElements.push({ ...link, element: path });
  }

  const splitLabel = (label) => {
    const words = label.split(" ");
    const lines = [""];
    for (const word of words) {
      const current = lines[lines.length - 1];
      if (current && `${current} ${word}`.length > 22) lines.push(word);
      else lines[lines.length - 1] = current ? `${current} ${word}` : word;
    }
    return lines;
  };

  const resetHighlight = () => {
    for (const element of nodeElements.values()) {
      element.classList.remove("is-muted", "is-active");
    }
    for (const link of linkElements) {
      link.element.classList.remove("is-muted", "is-active");
    }
    for (const card of cards) card.classList.remove("is-active");
  };

  const highlight = (id) => {
    const connected = new Set([id]);
    for (const link of linkElements) {
      if (link.source === id || link.target === id) {
        connected.add(link.source);
        connected.add(link.target);
      }
    }

    for (const [nodeId, element] of nodeElements) {
      element.classList.toggle("is-active", connected.has(nodeId));
      element.classList.toggle("is-muted", !connected.has(nodeId));
    }
    for (const link of linkElements) {
      const active = link.source === id || link.target === id;
      link.element.classList.toggle("is-active", active);
      link.element.classList.toggle("is-muted", !active);
    }
    for (const card of cards) {
      card.classList.toggle("is-active", card.dataset.project === id);
    }
  };

  for (const node of data.nodes) {
    const isProject = node.type === "project";
    const group = createSvgElement(isProject ? "a" : "g", {
      class: `graph-map-node graph-map-${node.type}`,
      "data-node": node.id,
      "aria-label": node.label
    });
    if (isProject) group.setAttribute("href", node.url);
    else group.setAttribute("tabindex", "0");

    const radius = node.id === "neural-networks" ? 27 : isProject ? 21 : 16;
    group.append(
      createSvgElement("circle", { cx: node.x, cy: node.y, r: radius + 11, class: "graph-node-halo" }),
      createSvgElement("circle", { cx: node.x, cy: node.y, r: radius, class: "graph-node-core" })
    );

    const label = createSvgElement("text", {
      x: node.x,
      y: node.y + radius + 30,
      "text-anchor": "middle",
      class: "graph-node-label"
    });
    splitLabel(node.label).forEach((line, index) => {
      const span = createSvgElement("tspan", { x: node.x, dy: index ? 18 : 0 });
      span.textContent = line;
      label.append(span);
    });
    group.append(label);

    const title = createSvgElement("title");
    title.textContent = isProject ? `Open project: ${node.label}` : `Topic: ${node.label}`;
    group.append(title);
    group.addEventListener("mouseenter", () => highlight(node.id));
    group.addEventListener("mouseleave", resetHighlight);
    group.addEventListener("focus", () => highlight(node.id));
    group.addEventListener("blur", resetHighlight);

    nodesLayer.append(group);
    nodeElements.set(node.id, group);
  }

  for (const card of cards) {
    card.addEventListener("mouseenter", () => highlight(card.dataset.project));
    card.addEventListener("mouseleave", resetHighlight);
    card.addEventListener("focus", () => highlight(card.dataset.project));
    card.addEventListener("blur", resetHighlight);
  }
})();
