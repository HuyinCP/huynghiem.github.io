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
  const viewBox = svg.viewBox.baseVal;
  const centerX = viewBox && viewBox.width ? viewBox.x + viewBox.width / 2 : 500;
  const centerY = viewBox && viewBox.height ? viewBox.y + viewBox.height / 2 : 310;
  const minBoundX = viewBox && viewBox.width ? viewBox.x + 25 : 45;
  const maxBoundX = viewBox && viewBox.width ? viewBox.x + viewBox.width - 25 : 955;
  const minBoundY = viewBox && viewBox.height ? viewBox.y + 25 : 45;
  const maxBoundY = viewBox && viewBox.height ? viewBox.y + viewBox.height - 45 : 585;

  const avgX = data.nodes.reduce((sum, node) => sum + node.x, 0) / Math.max(data.nodes.length, 1);
  const avgY = data.nodes.reduce((sum, node) => sum + node.y, 0) / Math.max(data.nodes.length, 1);
  for (const node of data.nodes) {
    node.x = node.x - avgX + centerX;
    node.y = node.y - avgY + centerY;
    node.vx = 0;
    node.vy = 0;
  }

  const nodesById = new Map(data.nodes.map((node) => [node.id, node]));
  const nodeElements = new Map();
  const linkElements = [];
  let dragState = null;
  let simulationFrame = null;
  let panState = null;
  let panX = 0;
  let panY = 0;
  const projectCount = data.nodes.filter((node) => node.type === "project").length;
  const topicCount = data.nodes.filter((node) => node.type === "topic").length;
  const projectCountElement = document.getElementById("graph-project-count");
  const topicCountElement = document.getElementById("graph-topic-count");
  if (projectCountElement) {
    projectCountElement.textContent = `${projectCount} ${projectCount === 1 ? "project" : "projects"}`;
  }
  if (topicCountElement) {
    topicCountElement.textContent = `${topicCount} ${topicCount === 1 ? "topic" : "topics"}`;
  }

  const createSvgElement = (name, attributes = {}) => {
    const element = document.createElementNS(namespace, name);
    for (const [key, value] of Object.entries(attributes)) {
      element.setAttribute(key, value);
    }
    return element;
  };

  const canvas = createSvgElement("g", { class: "graph-canvas" });
  const linksLayer = createSvgElement("g", { class: "graph-links" });
  const nodesLayer = createSvgElement("g", { class: "graph-map-nodes" });
  canvas.append(linksLayer, nodesLayer);
  svg.append(canvas);

  const updateCanvasPosition = () => {
    canvas.setAttribute("transform", `translate(${panX} ${panY})`);
  };

  const curvePath = (sourceNode, targetNode, offset = 0) => {
    const dx = targetNode.x - sourceNode.x;
    const dy = targetNode.y - sourceNode.y;
    const length = Math.max(Math.hypot(dx, dy), 1);
    const midpointX = (sourceNode.x + targetNode.x) / 2 - (dy / length) * offset;
    const midpointY = (sourceNode.y + targetNode.y) / 2 + (dx / length) * offset;
    return `M ${sourceNode.x} ${sourceNode.y} Q ${midpointX} ${midpointY} ${targetNode.x} ${targetNode.y}`;
  };

  for (const [index, link] of data.links.entries()) {
    const sourceNode = nodesById.get(link.source);
    const targetNode = nodesById.get(link.target);
    if (!sourceNode || !targetNode) continue;

    const path = createSvgElement("path", {
      d: curvePath(sourceNode, targetNode, (index % 3 - 1) * 12),
      class: "graph-link"
    });
    linksLayer.append(path);
    linkElements.push({ ...link, element: path, curveOffset: (index % 3 - 1) * 12 });
  }

  const splitLabel = (label) => {
    const words = label.split(" ");
    const lines = [""];
    for (const word of words) {
      const current = lines[lines.length - 1];
      if (current && `${current} ${word}`.length > 18) lines.push(word);
      else lines[lines.length - 1] = current ? `${current} ${word}` : word;
    }
    return lines;
  };

  const updateGraph = () => {
    for (const link of linkElements) {
      const sourceNode = nodesById.get(link.source);
      const targetNode = nodesById.get(link.target);
      link.element.setAttribute("d", curvePath(sourceNode, targetNode, link.curveOffset));
    }

    for (const node of data.nodes) {
      const group = nodeElements.get(node.id);
      if (!group) continue;
      const circles = group.querySelectorAll("circle");
      circles.forEach((circle) => {
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
      });
      const label = group.querySelector("text");
      label.setAttribute("x", node.x);
      label.setAttribute("y", node.y + (node.type === "project" ? 42 : node.id === "neural-networks" ? 48 : 39));
      label.querySelectorAll("tspan").forEach((span) => span.setAttribute("x", node.x));
    }
  };

  const startSimulation = () => {
    if (simulationFrame) return;

    let idleFrames = 0;
    const maxSpeed = 2.4;

    const tick = () => {
      const forces = new Map(data.nodes.map((node) => [node.id, { x: 0, y: 0 }]));

      for (let i = 0; i < data.nodes.length; i += 1) {
        for (let j = i + 1; j < data.nodes.length; j += 1) {
          const a = data.nodes[i];
          const b = data.nodes[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const distance = Math.max(Math.hypot(dx, dy), 24);
          const strength = 1250 / (distance * distance);
          const fx = (dx / distance) * strength;
          const fy = (dy / distance) * strength;
          forces.get(a.id).x -= fx;
          forces.get(a.id).y -= fy;
          forces.get(b.id).x += fx;
          forces.get(b.id).y += fy;
        }
      }

      for (const link of linkElements) {
        const sourceNode = nodesById.get(link.source);
        const targetNode = nodesById.get(link.target);
        const dx = targetNode.x - sourceNode.x;
        const dy = targetNode.y - sourceNode.y;
        const distance = Math.max(Math.hypot(dx, dy), 1);
        const stretch = (distance - 190) * 0.0038;
        const fx = (dx / distance) * stretch;
        const fy = (dy / distance) * stretch;
        forces.get(sourceNode.id).x += fx;
        forces.get(sourceNode.id).y += fy;
        forces.get(targetNode.id).x -= fx;
        forces.get(targetNode.id).y -= fy;
      }

      let maxMotion = 0;

      for (const node of data.nodes) {
        const force = forces.get(node.id);
        if (dragState?.node === node) {
          node.vx = 0;
          node.vy = 0;
          continue;
        }

        const centerPullX = (centerX - node.x) * 0.0014;
        const centerPullY = (centerY - node.y) * 0.0014;

        node.vx = (node.vx + force.x + centerPullX) * 0.78;
        node.vy = (node.vy + force.y + centerPullY) * 0.78;

        const speed = Math.hypot(node.vx, node.vy);
        if (speed > maxSpeed) {
          node.vx = (node.vx / speed) * maxSpeed;
          node.vy = (node.vy / speed) * maxSpeed;
        }

        const prevX = node.x;
        const prevY = node.y;
        node.x += node.vx;
        node.y += node.vy;
        maxMotion = Math.max(maxMotion, Math.hypot(node.x - prevX, node.y - prevY));
      }

      // Gentle positional bounding-box separation so node labels never overlap
      for (let pass = 0; pass < 2; pass += 1) {
        for (let i = 0; i < data.nodes.length; i += 1) {
          for (let j = i + 1; j < data.nodes.length; j += 1) {
            const a = data.nodes[i];
            const b = data.nodes[j];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const overlapX = a.halfW + b.halfW - Math.abs(dx);
            const overlapY =
              Math.min(a.y + a.bottomExtent, b.y + b.bottomExtent) -
              Math.max(a.y - a.topExtent, b.y - b.topExtent);

            if (overlapX > 0 && overlapY > 0) {
              const aPinned = dragState?.node === a;
              const bPinned = dragState?.node === b;
              const signX = dx === 0 ? (i % 2 === 0 ? 1 : -1) : Math.sign(dx);
              const signY = dy === 0 ? (j % 2 === 0 ? 1 : -1) : Math.sign(dy);

              if (overlapX < overlapY) {
                const sep = Math.min(overlapX * 0.22, 2.0) * signX;
                if (aPinned) {
                  b.x += sep * 2;
                } else if (bPinned) {
                  a.x -= sep * 2;
                } else {
                  a.x -= sep;
                  b.x += sep;
                }
                maxMotion = Math.max(maxMotion, Math.abs(sep));
              } else {
                const sep = Math.min(overlapY * 0.22, 2.0) * signY;
                if (aPinned) {
                  b.y += sep * 2;
                } else if (bPinned) {
                  a.y -= sep * 2;
                } else {
                  a.y -= sep;
                  b.y += sep;
                }
                maxMotion = Math.max(maxMotion, Math.abs(sep));
              }
            }
          }
        }
      }

      // Slowly glide the cluster's center of mass back to the center of the board when released
      if (!dragState && data.nodes.length > 0) {
        const meanX = data.nodes.reduce((sum, node) => sum + node.x, 0) / data.nodes.length;
        const meanY = data.nodes.reduce((sum, node) => sum + node.y, 0) / data.nodes.length;
        const rawShiftX = (centerX - meanX) * 0.015;
        const rawShiftY = (centerY - meanY) * 0.015;
        const shiftMag = Math.hypot(rawShiftX, rawShiftY);
        const clampedMag = Math.min(shiftMag, 1.4);
        const shiftX = shiftMag > 0 ? (rawShiftX / shiftMag) * clampedMag : 0;
        const shiftY = shiftMag > 0 ? (rawShiftY / shiftMag) * clampedMag : 0;

        if (shiftMag > 0.02) {
          for (const node of data.nodes) {
            node.x += shiftX;
            node.y += shiftY;
          }
          maxMotion = Math.max(maxMotion, clampedMag);
        }
      }

      for (const node of data.nodes) {
        if (dragState?.node === node) continue;
        const nodeMinX = minBoundX + node.halfW - 12;
        const nodeMaxX = maxBoundX - node.halfW + 12;
        const nodeMinY = minBoundY + node.topExtent - 10;
        const nodeMaxY = maxBoundY - node.bottomExtent + 12;
        node.x = Math.max(nodeMinX, Math.min(nodeMaxX, node.x));
        node.y = Math.max(nodeMinY, Math.min(nodeMaxY, node.y));
      }

      if (!panState && (Math.abs(panX) > 0.1 || Math.abs(panY) > 0.1)) {
        panX *= 0.94;
        panY *= 0.94;
        if (Math.abs(panX) <= 0.1) panX = 0;
        if (Math.abs(panY) <= 0.1) panY = 0;
        updateCanvasPosition();
        maxMotion = Math.max(maxMotion, Math.hypot(panX, panY));
      }

      updateGraph();

      if (!dragState && !panState && maxMotion < 0.04) {
        idleFrames += 1;
        if (idleFrames >= 14) {
          simulationFrame = null;
          return;
        }
      } else {
        idleFrames = 0;
      }

      simulationFrame = requestAnimationFrame(tick);
    };

    simulationFrame = requestAnimationFrame(tick);
  };

  const pointerPosition = (event) => {
    const ctm = canvas.getScreenCTM();
    if (ctm) {
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
      return { x: point.x, y: point.y };
    }
    const bounds = svg.getBoundingClientRect();
    return {
      x: ((event.clientX - bounds.left) / bounds.width) * 1000,
      y: ((event.clientY - bounds.top) / bounds.height) * 620
    };
  };

  const resetHighlight = () => {
    for (const element of nodeElements.values()) {
      element.classList.remove("is-muted", "is-active");
    }
    for (const link of linkElements) {
      link.element.classList.remove("is-muted", "is-active");
    }
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
  };

  for (const node of data.nodes) {
    const isProject = node.type === "project";
    const group = createSvgElement("g", {
      class: `graph-map-node graph-map-${node.type}`,
      "data-node": node.id,
      "aria-label": node.label,
      tabindex: "0"
    });

    const radius = node.id === "neural-networks" ? 18 : isProject ? 12 : 9;
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
    const lines = splitLabel(node.label);
    lines.forEach((line, index) => {
      const span = createSvgElement("tspan", { x: node.x, dy: index ? 18 : 0 });
      span.textContent = line;
      label.append(span);
    });
    group.append(label);

    const maxLineLength = Math.max(...lines.map((line) => line.length));
    const estimatedLabelWidth = maxLineLength * 8.4;
    const labelOffsetY = node.type === "project" ? 42 : node.id === "neural-networks" ? 48 : 39;
    node.halfW = Math.max(radius + 16, estimatedLabelWidth / 2 + 16);
    node.topExtent = radius + 16;
    node.bottomExtent = labelOffsetY + (lines.length - 1) * 18 + 14;

    const title = createSvgElement("title");
    title.textContent = isProject ? `Project: ${node.label}` : `Topic: ${node.label}`;
    group.append(title);
    group.addEventListener("mouseenter", () => highlight(node.id));
    group.addEventListener("mouseleave", resetHighlight);
    group.addEventListener("focus", () => highlight(node.id));
    group.addEventListener("blur", resetHighlight);
    group.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      dragState = { node };
      startSimulation();
      group.setPointerCapture(event.pointerId);
      group.classList.add("is-dragging");
    });
    group.addEventListener("pointermove", (event) => {
      if (!dragState || dragState.node !== node) return;
      const position = pointerPosition(event);
      node.x = Math.max(minBoundX, Math.min(maxBoundX, position.x));
      node.y = Math.max(minBoundY, Math.min(maxBoundY, position.y));
      updateGraph();
      startSimulation();
    });
    const finishNodeDrag = (event) => {
      if (!dragState || dragState.node !== node) return;
      dragState = null;
      group.classList.remove("is-dragging");
      if (group.hasPointerCapture(event.pointerId)) {
        group.releasePointerCapture(event.pointerId);
      }
      startSimulation();
    };
    group.addEventListener("pointerup", finishNodeDrag);
    group.addEventListener("pointercancel", finishNodeDrag);

    nodesLayer.append(group);
    nodeElements.set(node.id, group);

    // Refine bounding box using actual rendered SVG text metrics when available
    try {
      const labelBox = label.getBBox();
      if (labelBox && labelBox.width > 0) {
        node.halfW = Math.max(radius + 18, labelBox.width / 2 + 16);
        node.bottomExtent = Math.max(node.bottomExtent, labelBox.y + labelBox.height - node.y + 10);
      }
    } catch {
      // Keep estimated label metrics if SVG is hidden during initial render
    }
  }

  updateGraph();
  startSimulation();

  svg.addEventListener("pointerdown", (event) => {
    if (event.target !== svg) return;
    panState = { clientX: event.clientX, clientY: event.clientY, startX: panX, startY: panY };
    svg.setPointerCapture(event.pointerId);
    svg.classList.add("is-panning");
  });

  svg.addEventListener("pointermove", (event) => {
    if (!panState) return;
    const ctm = svg.getScreenCTM();
    if (ctm && ctm.a && ctm.d) {
      panX = panState.startX + (event.clientX - panState.clientX) / ctm.a;
      panY = panState.startY + (event.clientY - panState.clientY) / ctm.d;
    } else {
      const bounds = svg.getBoundingClientRect();
      panX = panState.startX + ((event.clientX - panState.clientX) / bounds.width) * 1000;
      panY = panState.startY + ((event.clientY - panState.clientY) / bounds.height) * 620;
    }
    updateCanvasPosition();
  });

  const finishPan = (event) => {
    if (!panState) return;
    panState = null;
    svg.classList.remove("is-panning");
    if (svg.hasPointerCapture(event.pointerId)) {
      svg.releasePointerCapture(event.pointerId);
    }
    startSimulation();
  };

  svg.addEventListener("pointerup", finishPan);
  svg.addEventListener("pointercancel", finishPan);
})();
