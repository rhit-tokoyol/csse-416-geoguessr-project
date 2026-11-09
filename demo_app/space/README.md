---
title: Where Is This?
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Guess where a GeoGuessr screenshot was taken
---

# Where is this?

A class project for CSSE 416 at Rose-Hulman. Paste a GeoGuessr screenshot and
see where our models would guess.

- **CLIP model:** frozen CLIP ViT-B/32 image features with a trained head that
  picks one of 1,905 map cells. A second head checks which side of the road
  traffic drives on by comparing the image with its mirror image, and that
  adjusts the cell guess.
- **ResNet model** (when included): frozen ResNet-101 features with a trained
  head that picks a country.

Trained on street-level images from the
[OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) dataset.

Uploaded screenshots are only used to make the guess. They are not saved.

For screenshots only: please don't use this during live GeoGuessr games.
