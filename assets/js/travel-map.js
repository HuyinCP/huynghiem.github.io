const travelMapElement = document.querySelector("#travel-map");

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

  const destinations = [
    {
      name: "Da Nang",
      coordinates: [108.2062, 16.0471],
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
      name: "Hanoi",
      coordinates: [105.8522, 21.0285],
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
      name: "Singapore",
      coordinates: [103.8519, 1.2903],
      visits: [
        {
          date: "March 2025",
          url: "travel/singapore-march-2025.html",
        },
      ],
    },
    {
      name: "Ho Chi Minh City",
      coordinates: [106.7008, 10.7769],
      visits: [
        {
          date: "December 2025",
          url: "travel/ho-chi-minh-city-december-2025.html",
        },
      ],
    },
    {
      name: "Taipei",
      coordinates: [121.5645, 25.034],
      visits: [
        {
          date: "March 2026",
          url: "travel/taiwan-march-2026.html",
        },
      ],
    },
    {
      name: "Taoyuan",
      coordinates: [121.301, 24.9936],
      visits: [
        {
          date: "March 2026",
          url: "travel/taiwan-march-2026.html",
        },
      ],
    },
  ];

  const mapBounds = new maplibregl.LngLatBounds();

  for (const destination of destinations) {
    const visitLinks = destination.visits
      .map(
        (visit) =>
          `<li><a href="${visit.url}">${visit.date}</a></li>`,
      )
      .join("");

    const popup = new maplibregl.Popup({
      offset: 24,
      maxWidth: "260px",
    }).setHTML(`
      <div class="travel-map-popup">
        <h3>${destination.name}</h3>
        <ul>${visitLinks}</ul>
      </div>
    `);

    const markerElement = document.createElement("button");
    markerElement.className = "travel-marker";
    markerElement.type = "button";
    markerElement.setAttribute("aria-label", `View trips to ${destination.name}`);
    markerElement.title = destination.name;

    const markerPin = document.createElement("span");
    markerPin.className = "travel-marker-pin";
    markerPin.setAttribute("aria-hidden", "true");
    markerElement.append(markerPin);

    new maplibregl.Marker({
      element: markerElement,
      anchor: "bottom",
    })
      .setLngLat(destination.coordinates)
      .setPopup(popup)
      .addTo(travelMap);

    mapBounds.extend(destination.coordinates);
  }

  travelMap.once("load", () => {
    travelMap.fitBounds(mapBounds, {
      padding: 58,
      maxZoom: 5,
      duration: 900,
    });
  });
}
