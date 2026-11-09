const API = "http://127.0.0.1:8000/predict";

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
        target: { tabId: tab.id }, world: "MAIN", func: placePin, args: [out.lat, out.lon, 5],
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