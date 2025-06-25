# FFItemRecognition

Detect items from images and set up shop!

This repository contains two projects:
- Python-based item embedding and setup scripts in `src/`.
- A React + Vite static Web App hosted on Cloudflare Pages called [Retrobution Shoplist](https://retrobution-shoplist.pages.dev) in `item-recognition-web/`.

## Setup

```sh
pip install -r requirements.txt
```

Also obtain a Google Service Accouunt with Google Drive and Google Sheets API enabled, and save the credentials at `resources/google-service-account.json` for Price Guide access. If you do not, price information will be absent in the label directory, but you can still run the scripts.

## Building Embeddings, Labels, Model

This is a necessary step before testing or running the web app.

```sh
python src/prepare_embeds_model.py
```

## Visualization

To give the model a bunch of images and see the results for each image, put your images in `resources/test_images` and run the following script.

```sh
python src/detect_items.py
```

The results will be drawn into `resources/output/test_images` and the coordinates / item matches will be available as JSON files in `resources/output/test_images_json`.

