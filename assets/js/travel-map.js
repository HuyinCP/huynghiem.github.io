const travelMapElement = document.querySelector("#travel-map");

function createCurvedArc(startLngLat, endLngLat, curvature = 0.22, segments = 64) {
  const startMercator = maplibregl.MercatorCoordinate.fromLngLat(startLngLat);
  const endMercator = maplibregl.MercatorCoordinate.fromLngLat(endLngLat);

  const dx = endMercator.x - startMercator.x;
  const dy = endMercator.y - startMercator.y;
  const controlX = (startMercator.x + endMercator.x) * 0.5 + curvature * dy;
  const controlY = (startMercator.y + endMercator.y) * 0.5 - curvature * dx;

  const coordinates = [];

  for (let step = 0; step <= segments; step += 1) {
    const t = step / segments;
    const oneMinusT = 1 - t;
    const x =
      oneMinusT * oneMinusT * startMercator.x +
      2 * oneMinusT * t * controlX +
      t * t * endMercator.x;
    const y =
      oneMinusT * oneMinusT * startMercator.y +
      2 * oneMinusT * t * controlY +
      t * t * endMercator.y;
    const point = new maplibregl.MercatorCoordinate(x, y, 0).toLngLat();
    coordinates.push([point.lng, point.lat]);
  }

  return coordinates;
}

function sampleArcPoint(coordinates, progress) {
  const clamped = Math.max(0, Math.min(1, progress));
  const scaledIndex = clamped * (coordinates.length - 1);
  const lowerIndex = Math.floor(scaledIndex);
  const upperIndex = Math.min(coordinates.length - 1, lowerIndex + 1);
  const fraction = scaledIndex - lowerIndex;

  const start = coordinates[lowerIndex];
  const end = coordinates[upperIndex];

  return [
    start[0] + (end[0] - start[0]) * fraction,
    start[1] + (end[1] - start[1]) * fraction,
  ];
}

if (travelMapElement) {
  const travelMap = new maplibregl.Map({
    container: travelMapElement,
    style: "https://tiles.openfreemap.org/styles/bright",
    center: [108.5, 15],
    zoom: 3.5,
    attributionControl: false,
  });

  travelMap.scrollZoom.disable();
  travelMap.addControl(
    new maplibregl.NavigationControl({
      showCompass: false,
      visualizePitch: false,
    }),
    "top-right",
  );
  travelMap.addControl(
    new maplibregl.AttributionControl({
      compact: true,
    }),
    "bottom-right",
  );

  const origin = {
    id: "ho-chi-minh-city",
    name: "Ho Chi Minh City",
    coordinates: [106.7008, 10.7769],
    isOrigin: true,
    visits: [
      {
        date: "December 2025",
        url: "travel/ho-chi-minh-city-december-2025.html",
      },
    ],
  };

  const outboundDestinations = [
    {
      id: "da-nang",
      name: "Da Nang",
      coordinates: [108.2062, 16.0471],
      curvature: 0.24,
      visits: [
        {
          date: "August 2024",
          url: "travel/da-nang-august-2024.html",
        },
        {
          date: "December 2026",
          url: "travel/da-nang-december-2026.html",
        },
      ],
    },
    {
      id: "hanoi",
      name: "Hanoi",
      coordinates: [105.8522, 21.0285],
      curvature: 0.28,
      visits: [
        {
          date: "December 2024",
          url: "travel/hanoi-december-2024.html",
        },
        {
          date: "July 2026",
          url: "travel/hanoi-july-2026.html",
        },
      ],
    },
    {
      id: "singapore",
      name: "Singapore",
      coordinates: [103.8519, 1.2903],
      curvature: -0.26,
      visits: [
        {
          date: "March 2025",
          url: "travel/singapore-march-2025.html",
        },
      ],
    },
    {
      id: "taipei",
      name: "Taipei",
      coordinates: [121.5645, 25.034],
      curvature: 0.17,
      visits: [
        {
          date: "March 2026",
          url: "travel/taiwan-march-2026.html",
        },
      ],
    },
    {
      id: "taoyuan",
      name: "Taoyuan",
      coordinates: [121.301, 24.9936],
      curvature: 0.25,
      visits: [
        {
          date: "March 2026",
          url: "travel/taiwan-march-2026.html",
        },
      ],
    },
  ];

  const allPlaces = [origin, ...outboundDestinations];
  const allOutboundIds = outboundDestinations.map((item) => item.id);
  const mapBounds = new maplibregl.LngLatBounds();

  const routeArcs = outboundDestinations.map((destination, index) => {
    const coordinates = createCurvedArc(
      origin.coordinates,
      destination.coordinates,
      destination.curvature,
      64,
    );

    for (const point of coordinates) {
      mapBounds.extend(point);
    }

    return {
      id: destination.id,
      name: destination.name,
      phaseOffset: index * 0.19,
      coordinates,
    };
  });

  const routeFeatureCollection = {
    type: "FeatureCollection",
    features: routeArcs.map((arc) => ({
      type: "Feature",
      properties: {
        id: arc.id,
        name: arc.name,
      },
      geometry: {
        type: "LineString",
        coordinates: arc.coordinates,
      },
    })),
  };

  const markerElementsById = new Map();
  let isMapLoaded = false;

  function setHighlightedRoutes(activeIds = []) {
    const activeSet = new Set(activeIds);
    const hasActiveSelection = activeSet.size > 0;

    for (const [placeId, element] of markerElementsById.entries()) {
      if (placeId === origin.id) {
        element.classList.toggle("is-active", hasActiveSelection);
      } else {
        element.classList.toggle("is-active", activeSet.has(placeId));
      }
    }

    if (!isMapLoaded || !travelMap.getLayer("travel-routes-highlight")) {
      return;
    }

    travelMap.setFilter("travel-routes-highlight", [
      "in",
      ["get", "id"],
      ["literal", activeIds],
    ]);
    travelMap.setPaintProperty(
      "travel-routes-line",
      "line-opacity",
      hasActiveSelection ? 0.35 : 0.9,
    );
    travelMap.setPaintProperty(
      "travel-routes-glow",
      "line-opacity",
      hasActiveSelection ? 0.12 : 0.22,
    );
  }

  for (const place of allPlaces) {
    const visitLinks = place.visits
      .map((visit) => `<li><a href="${visit.url}">${visit.date}</a></li>`)
      .join("");
    const originBadge = place.isOrigin
      ? '<p class="travel-popup-badge">Origin</p>'
      : "";

    const popup = new maplibregl.Popup({
      offset: place.isOrigin ? 18 : 24,
      maxWidth: "260px",
    }).setHTML(`
      <div class="travel-map-popup">
        ${originBadge}
        <h3>${place.name}</h3>
        <ul>${visitLinks}</ul>
      </div>
    `);

    const markerElement = document.createElement("button");
    markerElement.className = place.isOrigin
      ? "travel-marker travel-marker--origin"
      : "travel-marker";
    markerElement.type = "button";
    markerElement.setAttribute(
      "aria-label",
      place.isOrigin
        ? `Origin: ${place.name}. View trips`
        : `View trips to ${place.name}`,
    );
    markerElement.title = place.isOrigin
      ? `Origin: ${place.name}`
      : place.name;

    if (place.isOrigin) {
      const markerHalo = document.createElement("span");
      markerHalo.className = "travel-marker-halo";
      markerHalo.setAttribute("aria-hidden", "true");

      const markerHub = document.createElement("span");
      markerHub.className = "travel-marker-hub";
      markerHub.setAttribute("aria-hidden", "true");

      markerElement.append(markerHalo, markerHub);
    } else {
      const markerPin = document.createElement("span");
      markerPin.className = "travel-marker-pin";
      markerPin.setAttribute("aria-hidden", "true");
      markerElement.append(markerPin);
    }

    const highlightTargetIds = place.isOrigin ? allOutboundIds : [place.id];

    markerElement.addEventListener("mouseenter", () => {
      setHighlightedRoutes(highlightTargetIds);
    });
    markerElement.addEventListener("mouseleave", () => {
      setHighlightedRoutes([]);
    });
    markerElement.addEventListener("focus", () => {
      setHighlightedRoutes(highlightTargetIds);
    });
    markerElement.addEventListener("blur", () => {
      setHighlightedRoutes([]);
    });

    new maplibregl.Marker({
      element: markerElement,
      anchor: place.isOrigin ? "center" : "bottom",
    })
      .setLngLat(place.coordinates)
      .setPopup(popup)
      .addTo(travelMap);

    markerElementsById.set(place.id, markerElement);
    mapBounds.extend(place.coordinates);
  }

  const entryLinks = document.querySelectorAll(".travel-list .travel-entry");
  for (const entryLink of entryLinks) {
    const entryHref = entryLink.getAttribute("href");
    if (!entryHref) {
      continue;
    }

    const matchingPlaces = allPlaces.filter((place) =>
      place.visits.some((visit) => visit.url === entryHref),
    );
    if (!matchingPlaces.length) {
      continue;
    }

    const targetIds = matchingPlaces.some((place) => place.isOrigin)
      ? allOutboundIds
      : matchingPlaces.map((place) => place.id);

    entryLink.addEventListener("mouseenter", () => {
      setHighlightedRoutes(targetIds);
    });
    entryLink.addEventListener("mouseleave", () => {
      setHighlightedRoutes([]);
    });
    entryLink.addEventListener("focus", () => {
      setHighlightedRoutes(targetIds);
    });
    entryLink.addEventListener("blur", () => {
      setHighlightedRoutes([]);
    });
  }

  function buildPulseFeatureCollection(normalizedTime) {
    return {
      type: "FeatureCollection",
      features: routeArcs.map((arc) => {
        const progress = (normalizedTime + arc.phaseOffset) % 1;
        const envelope = Math.sin(Math.PI * progress);

        return {
          type: "Feature",
          properties: {
            id: arc.id,
            envelope,
          },
          geometry: {
            type: "Point",
            coordinates: sampleArcPoint(arc.coordinates, progress),
          },
        };
      }),
    };
  }

  travelMap.once("load", () => {
    isMapLoaded = true;

    travelMap.addSource("travel-routes", {
      type: "geojson",
      lineMetrics: true,
      data: routeFeatureCollection,
    });

    travelMap.addLayer({
      id: "travel-routes-glow",
      type: "line",
      source: "travel-routes",
      layout: {
        "line-cap": "round",
        "line-join": "round",
      },
      paint: {
        "line-color": "#9b77bd",
        "line-width": ["interpolate", ["linear"], ["zoom"], 3, 6, 7, 10],
        "line-opacity": 0.22,
        "line-blur": 3,
      },
    });

    travelMap.addLayer({
      id: "travel-routes-line",
      type: "line",
      source: "travel-routes",
      layout: {
        "line-cap": "round",
        "line-join": "round",
      },
      paint: {
        "line-gradient": [
          "interpolate",
          ["linear"],
          ["line-progress"],
          0,
          "rgba(88, 53, 128, 0.96)",
          0.55,
          "rgba(131, 92, 174, 0.88)",
          1,
          "rgba(155, 119, 189, 0.72)",
        ],
        "line-width": ["interpolate", ["linear"], ["zoom"], 3, 2.2, 7, 3.4],
        "line-opacity": 0.9,
      },
    });

    travelMap.addLayer({
      id: "travel-routes-highlight",
      type: "line",
      source: "travel-routes",
      filter: ["in", ["get", "id"], ["literal", []]],
      layout: {
        "line-cap": "round",
        "line-join": "round",
      },
      paint: {
        "line-color": "#5d3988",
        "line-width": ["interpolate", ["linear"], ["zoom"], 3, 3.4, 7, 4.8],
        "line-opacity": 0.98,
      },
    });

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (!prefersReducedMotion) {
      travelMap.addSource("travel-route-pulses", {
        type: "geojson",
        data: buildPulseFeatureCollection(0),
      });

      travelMap.addLayer({
        id: "travel-route-pulses-glow",
        type: "circle",
        source: "travel-route-pulses",
        paint: {
          "circle-radius": [
            "+",
            3,
            ["*", 5, ["get", "envelope"]],
          ],
          "circle-color": "#9b77bd",
          "circle-opacity": ["*", 0.35, ["get", "envelope"]],
          "circle-blur": 0.6,
        },
      });

      travelMap.addLayer({
        id: "travel-route-pulses-dot",
        type: "circle",
        source: "travel-route-pulses",
        paint: {
          "circle-radius": [
            "+",
            1.6,
            ["*", 2.4, ["get", "envelope"]],
          ],
          "circle-color": "#ffffff",
          "circle-stroke-width": 1.6,
          "circle-stroke-color": "#654386",
          "circle-opacity": ["*", 0.95, ["get", "envelope"]],
          "circle-stroke-opacity": ["*", 0.95, ["get", "envelope"]],
        },
      });

      const cycleDurationMs = 3800;
      const animatePulses = (timestamp) => {
        if (!document.hidden) {
          const pulseSource = travelMap.getSource("travel-route-pulses");
          if (pulseSource) {
            const normalizedTime = (timestamp % cycleDurationMs) / cycleDurationMs;
            pulseSource.setData(buildPulseFeatureCollection(normalizedTime));
          }
        }
        window.requestAnimationFrame(animatePulses);
      };

      window.requestAnimationFrame(animatePulses);
    }

    travelMap.fitBounds(mapBounds, {
      padding: 58,
      maxZoom: 5,
      duration: 900,
    });
  });
}
