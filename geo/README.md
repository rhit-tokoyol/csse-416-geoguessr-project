Local server for the browser extension (extension/). The extension posts a screenshot to /predict.

    cd geo
    uvicorn server:app --host 127.0.0.1 --port 8000

Needs two files in this folder, from the same training run:
- the trained head, `finetuned.pt` (or point HEAD_PATH at it), a raw head saved by main.ipynb: the residual MLP or the plain MLP
- `centres.npz`, with `lat`, `lon` and `classes`. Make it from train.csv:
  `python make_centres.py /work/.../OpenWorld/train.csv centres.npz <head.pt>`
  (the optional head argument checks that both have the same number of cells)

Settings, as environment variables:
- `CLIP_MODEL`: the CLIP backbone the head was trained on, default `openai/clip-vit-base-patch32`.
  For an L/14 head: `CLIP_MODEL=openai/clip-vit-large-patch14`
- `HEAD_PATH`: the head file, default `finetuned.pt`
- `TORCH_THREADS`: CPU threads, default 2. L/14 is much slower than B/32 on a CPU, so try 4 or more.

Example (PowerShell), L/14 residual head:

    $env:CLIP_MODEL = "openai/clip-vit-large-patch14"
    $env:HEAD_PATH = "train_cell_sampler_14.pt"
    $env:TORCH_THREADS = "6"
    uvicorn server:app --host 127.0.0.1 --port 8000

The first start downloads CLIP (L/14 is a few GB). The guess is the cell with the best expected GeoGuessr score,
not just the most probable one.

## Other game sites (OpenGuessr, ...)

The extension works on any site: click the icon (or Alt+G) and it screenshots the tab, asks the server and shows the
guess in a box with a Google Maps link. Placing the pin on the game's own guess map needs site-specific code:
- geoguessr.com has its own (`placePin`).
- every other site uses `placePinGeneric` in `extension/background.js`. It looks for a Leaflet, MapLibre/Mapbox or
  Google Maps object (on `window`, or reachable from a map element's React components), moves the map and clicks
  the spot. If it finds none, the box says what it looked at (for example `map instance not found (looked at .leaflet-container)`).

If the pin isn't placed on a site, open DevTools (F12) on the game page and paste these into the console, then use the output
to add a case for that site:

    [...document.querySelectorAll('[class*="map"], .leaflet-container, .gm-style, .mapboxgl-map, .maplibregl-map')].map(e => e.tagName + "." + e.className)
    Object.keys(window).filter(k => /map/i.test(k))
    typeof L + " " + typeof maplibregl + " " + typeof mapboxgl + " " + typeof google
