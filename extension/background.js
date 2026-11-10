const API = "http://127.0.0.1:8000/predict";

// GeoGuessr has its own pin code (placePin). Other sites, such as OpenGuessr, use placePinGeneric,
// which looks for a map object and says what it found, so it can be adjusted per site.
const isGeoGuessr = (url) => { try { return /(^|\.)geoguessr\.com$/.test(new URL(url).hostname); } catch (e) { return false; } };

chrome.action.onClicked.addListener(async (tab) => {
  let out;
  try {
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "jpeg", quality: 90 });
    const blob = await (await fetch(dataUrl)).blob();
    const form = new FormData();
    form.append("file", blob, "shot.jpg");
    const res = await fetch(API, { method: "POST", body: form });
    out = await res.json();
  } catch (e) {
    out = { error: String(e) };
  }
  if (!out.error) {
    try {
      const r = await chrome.scripting.executeScript({
        target: { tabId: tab.id }, world: "MAIN", func: isGeoGuessr(tab.url) ? placePin : placePinGeneric, args: [out.lat, out.lon, 5],
      });
      out.pin = r[0].result;
    } catch (e) {
      out.pin = "failed: " + String(e);
    }
  }
  await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: showResult, args: [out] });
});

// runs inside the page: draws a small box with the result
function showResult(out) {
  let box = document.getElementById("geo-demo-box");
  if (!box) {
    box = document.createElement("div");
    box.id = "geo-demo-box";
    box.style.cssText =
      "position:fixed;top:10px;left:10px;z-index:2147483647;background:#111;color:#fff;" +
      "padding:10px 14px;font:14px sans-serif;border-radius:8px;max-width:320px;cursor:pointer";
    document.body.appendChild(box);
  }
  if (out.error) { box.textContent = "Error: " + out.error; return; }
  const link = "https://www.google.com/maps?q=" + out.lat + "," + out.lon;
  box.innerHTML =
    "Predicted: " + out.lat.toFixed(3) + ", " + out.lon.toFixed(3) +
    "<br>Cell: " + out.cell +
    "<br>pin: " + (out.pin || "") +
    '<br><a href="' + link + '" target="_blank" style="color:#8cf">open in Google Maps</a>' +
    "<br><small>click box to dismiss</small>";
  box.onclick = (e) => { if (e.target.tagName !== "A") box.remove(); };
}

// runs inside the page of any other game: finds a map object (Leaflet, MapLibre/Mapbox or Google Maps),
// moves it to the guess and clicks there. If it can't find one it returns what it looked at.
async function placePinGeneric(lat, lng, zoom) {
  const kind = (m) => !m || typeof m !== "object" ? null
    : typeof m.latLngToContainerPoint === "function" && typeof m.setView === "function" ? "leaflet"
    : typeof m.project === "function" && typeof m.jumpTo === "function" ? "mapbox"
    : typeof m.setCenter === "function" && typeof m.getZoom === "function" ? "google" : null;
  let found = null;
  for (const name of ["map", "guessMap", "gmap", "leafletMap", "guessmap"]) {   // a map object left on window
    const m = window[name];
    if (kind(m)) { found = { map: m, type: kind(m) }; break; }
  }
  const sels = [".leaflet-container", ".mapboxgl-map", ".maplibregl-map", ".gm-style",
                '[class*="guess"][class*="map"]', '[class*="map_"]', '[class*="map"]'];
  const tried = [];
  for (const sel of sels) {                    // or one reachable from a map element's React components
    if (found) break;
    for (const el of document.querySelectorAll(sel)) {
      tried.push(sel);
      const key = Object.keys(el).find(k => k.startsWith("__reactFiber$"));
      let fiber = key ? el[key] : null;
      for (let i = 0; fiber && i < 60 && !found; i++, fiber = fiber.return) {
        const p = fiber.memoizedProps;
        for (const c of [p && p.map, p && p.value && p.value.map, fiber.stateNode && fiber.stateNode.map]) {
          if (kind(c)) { found = { map: c, type: kind(c) }; break; }
        }
      }
      if (found) break;
    }
  }
  if (!found && window.L && window.L.Map && window.L.Map.prototype) {
    // Leaflet keeps no list of its maps. Catch one instead: wrap a few methods that every map calls when the
    // window is resized, fire a resize, then put the methods back.
    const proto = window.L.Map.prototype, names = ["getSize", "getCenter", "getZoom", "getPixelBounds"], orig = {}, seen = new Set();
    for (const n of names) {
      orig[n] = proto[n];
      if (typeof orig[n] === "function") proto[n] = function () { seen.add(this); return orig[n].apply(this, arguments); };
    }
    window.dispatchEvent(new Event("resize"));
    await new Promise((res) => setTimeout(res, 150));
    for (const n of names) if (orig[n]) proto[n] = orig[n];
    const maps = [...seen].filter((m) => kind(m) === "leaflet");
    const box = document.querySelector(".leaflet-container");
    if (maps.length) found = { map: maps.find((m) => m.getContainer && m.getContainer() === box) || maps[0], type: "leaflet" };
    else tried.push("(Leaflet is on the page, but no map answered a resize)");
  }
  if (!found) return "map instance not found (looked at " + (tried.length ? tried.join(" ") : "no map-like elements") + "; L is " + typeof window.L + ")";
  const { map, type } = found;
  if (type === "leaflet") {
    map.setView([lat, lng], zoom, { animate: false });
    const el = map.getContainer(), pt = map.latLngToContainerPoint([lat, lng]), r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) {      // a visible map: click it like a person would
      el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window, clientX: r.left + pt.x, clientY: r.top + pt.y }));
    } else {                                 // a hidden or collapsed map: fire Leaflet's own click event
      const ll = window.L && window.L.latLng ? window.L.latLng(lat, lng) : { lat: lat, lng: lng };
      map.fire("click", { latlng: ll, containerPoint: pt, layerPoint: map.latLngToLayerPoint([lat, lng]), originalEvent: new MouseEvent("click") });
    }
  } else if (type === "mapbox") {
    map.jumpTo({ center: [lng, lat], zoom: zoom });
    map.fire("click", { lngLat: { lng: lng, lat: lat }, point: map.project([lng, lat]), originalEvent: new MouseEvent("click") });
  } else {
    map.setCenter({ lat: lat, lng: lng });
    map.setZoom(zoom);
    google.maps.event.trigger(map, "click", { latLng: new google.maps.LatLng(lat, lng) });
  }
  return "pin placed (" + type + ")";
}

function placePin(lat, lng, zoom) {
  const el = document.querySelector('[class^="guess-map_canvas__"]');
  if (!el) return "guess map element not found";
  const key = Object.keys(el).find(k => k.startsWith("__reactFiber$"));
  if (!key) return "react fiber not found";
  let fiber = el[key], map = null;
  while (fiber && !map) {
    const p = fiber.memoizedProps;
    if (p && p.map && typeof p.map.setCenter === "function") map = p.map;
    fiber = fiber.return;
  }
  if (!map) return "map instance not found";
  map.setCenter({ lat: lat, lng: lng });
  map.setZoom(zoom);
  google.maps.event.trigger(map, "click", { latLng: new google.maps.LatLng(lat, lng) });
  return "pin placed";
}