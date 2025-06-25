import argparse
import asyncio
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import aiofiles
import cv2
import gspread
import httpx
import numpy as np
import torch
from torchvision import models, transforms
from tqdm import tqdm

REPO_RELEASE_URL = (
    "https://github.com/FinnHornhoover/FFInfoPacks/releases/latest/download"
)
PRICE_GUIDE_URL = "https://docs.google.com/spreadsheets/d/15ObDHwLa7rrd0b54RLJCJEsEIEtUHnEjtqV3kQBUCu4"


async def download_file(url: str, path: Path) -> None:
    """
    Downloads a file from a URL to a local path asynchronously.

    Parameters
    ----------
    url : str
        The URL to download the file from.
    path : Path
        The local path to save the file to.
    """
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=5),
        timeout=httpx.Timeout(None),
        follow_redirects=True,
    ) as client:
        async with client.stream("GET", url) as stream:
            stream.raise_for_status()

            async with aiofiles.open(path, "wb") as f:
                async for chunk in stream.aiter_bytes(chunk_size=(1 << 16)):
                    await f.write(chunk)


def construct_index_and_embedder(
    pack_dir_path: Path,
    resource_path: Path,
    output_embeddings_path: Path,
    output_model_path: Path,
    output_labels_path: Path,
    item_prices: dict[str, str],
) -> None:
    """
    Constructs the index and embedder for the icons.

    Parameters
    ----------
    pack_dir_path : Path
        The path to the pack directory, e.g. `retrobution_r4/`.
    resource_path : Path
        The path to the resource directory.
    output_embeddings_path : Path
        The path to the output embeddings directory.
    output_model_path : Path
        The path to the output model file.
    output_labels_path : Path
        The path to the output labels directory.
    item_prices : dict[str, str]
        A dictionary of item names and their prices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    # Remove classification head, we just want features
    model.classifier = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    # Export the embedder model to ONNX
    print(f"Exporting embedder model to {output_model_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(8, 3, 224, 224).to(device),
            output_model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=18,
        )

    # Preprocessing transforms
    preprocess_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and process templates
    template_embeddings = []
    template_label_ids = []

    def add_embedding(image: np.ndarray, template_ids: list[str]) -> None:
        """
        Adds an embedding to the list of embeddings.

        Parameters
        ----------
        image : np.ndarray
            The image to add the embedding to.
        template_ids : list[str]
            The template ID group to add the embedding to.
        """
        image_tensor = preprocess_transforms(image).unsqueeze(0).to(device)
        embedding = model(image_tensor).cpu().numpy()
        template_embeddings.append(embedding)
        template_label_ids.append(template_ids)

    background_img = cv2.imread(resource_path / "inv_empty.png", cv2.IMREAD_COLOR)
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    red_background_img = cv2.imread(
        resource_path / "inv_red_empty.png", cv2.IMREAD_COLOR
    )
    red_background_img = cv2.cvtColor(red_background_img, cv2.COLOR_BGR2RGB)
    bank_background_img = cv2.imread(resource_path / "bank_empty.png", cv2.IMREAD_COLOR)
    bank_background_img = cv2.cvtColor(bank_background_img, cv2.COLOR_BGR2RGB)
    selected_bank_background_img = cv2.imread(
        resource_path / "bank_empty_selected.png", cv2.IMREAD_COLOR
    )
    selected_bank_background_img = cv2.cvtColor(
        selected_bank_background_img, cv2.COLOR_BGR2RGB
    )
    null_img = np.zeros_like(background_img)

    with torch.no_grad():
        add_embedding(null_img, ["00::0000"])
        add_embedding(background_img, ["00::0000"])
        add_embedding(red_background_img, ["00::0000"])
        add_embedding(bank_background_img, ["00::0000"])
        add_embedding(selected_bank_background_img, ["00::0000"])

    with open(pack_dir_path / "info" / "item_info.json", "r") as f:
        item_info = json.load(f)

    truncated_item_info = {
        t: {
            "Type": d["DisplayType"],
            "Name": d["Name"],
            "Level": d["ContentLevel"],
            "Rarity": d["Rarity"],
            "Icon": d["Icon"].split("/")[-1],
            "Price": item_prices.get(d["Name"], ""),
        }
        for t, d in item_info.items()
    }

    image_info = defaultdict(list)
    for item in item_info.values():
        image_info[item["Icon"]].append(item)

    for icon_name in image_info:
        image_info[icon_name] = [
            item["ID"]
            for item in sorted(
                image_info[icon_name],
                key=lambda x: (
                    int(x["Obtainable"]),
                    int("Prototype" not in x["Name"]),
                    int("KND Hoverboard" not in x["Name"]),
                    x["RarityID"],
                    x["ContentLevel"],
                ),
                reverse=True,
            )
        ]

    with torch.no_grad():
        for icon_name, label_ids in tqdm(
            image_info.items(), desc="Embedding Icons", total=len(image_info)
        ):
            template_file = pack_dir_path / icon_name
            template_raw = cv2.imread(template_file, cv2.IMREAD_UNCHANGED)

            if template_raw is None or template_raw.shape[:2] != (64, 64):
                continue

            # Create mask from alpha channel and apply it
            alpha_mask = template_raw[:, :, 3] / 255.0
            template_rgb = cv2.cvtColor(template_raw[:, :, :3], cv2.COLOR_BGR2RGB)
            template_rgb[:, :, :] = template_rgb[:, :, :] * alpha_mask[:, :, np.newaxis]
            add_embedding(template_rgb, label_ids)

            # place the image inside the inv_empty.png and embed it
            template_w_background = background_img.copy()
            template_w_background[:, :, :] = (
                template_w_background[:, :, :] * (1 - alpha_mask[:, :, np.newaxis])
                + template_rgb[:, :, :]
            )
            add_embedding(template_w_background, label_ids)

            # place the image inside the inv_red_empty.png and embed it
            template_w_red_background = red_background_img.copy()
            template_w_red_background[:, :, :] = (
                template_w_red_background[:, :, :] * (1 - alpha_mask[:, :, np.newaxis])
                + template_rgb[:, :, :]
            )
            add_embedding(template_w_red_background, label_ids)

            # prepare bank images
            template_bgra = cv2.resize(
                template_raw, (56, 56), interpolation=cv2.INTER_AREA
            )
            shrinked_template_bgra = np.zeros_like(template_raw)
            shrinked_template_bgra[4:60, 4:60, :] = template_bgra
            padded_alpha_mask = shrinked_template_bgra[:, :, 3] / 255.0
            padded_template_rgb = cv2.cvtColor(
                shrinked_template_bgra[:, :, :3], cv2.COLOR_BGR2RGB
            )

            # place the image inside the bank_empty.png and embed it
            template_w_bank_background = bank_background_img.copy()
            template_w_bank_background[:, :, :] = (
                template_w_bank_background[:, :, :]
                * (1 - padded_alpha_mask[:, :, np.newaxis])
                + padded_template_rgb[:, :, :] * padded_alpha_mask[:, :, np.newaxis]
            )
            add_embedding(template_w_bank_background, label_ids)

            # place the image inside the bank_empty_selected.png and embed it
            template_w_selected_bank_background = selected_bank_background_img.copy()
            template_w_selected_bank_background[:, :, :] = (
                template_w_selected_bank_background[:, :, :]
                * (1 - padded_alpha_mask[:, :, np.newaxis])
                + padded_template_rgb[:, :, :] * padded_alpha_mask[:, :, np.newaxis]
            )
            add_embedding(template_w_selected_bank_background, label_ids)

    template_embeddings = np.vstack(template_embeddings)

    with open(output_labels_path / "item_info_truncated.json", "w") as f:
        json.dump(truncated_item_info, f)

    with open(output_labels_path / "item_label_ids.json", "w") as f:
        json.dump(template_label_ids, f)

    ranges = np.vstack(
        (np.min(template_embeddings, axis=0), np.max(template_embeddings, axis=0))
    )
    starts = ranges[0, :]
    steps = (ranges[1, :] - ranges[0, :]) / 255
    embeddings_quantized = np.uint8((template_embeddings - starts) / steps)

    with open(output_embeddings_path.with_suffix(".qvals.bin"), "wb") as f:
        f.write(embeddings_quantized.tobytes())

    with open(output_embeddings_path.with_suffix(".starts.bin"), "wb") as f:
        f.write(starts.tobytes())

    with open(output_embeddings_path.with_suffix(".steps.bin"), "wb") as f:
        f.write(steps.tobytes())


def construct_item_prices(google_service_account_json: str) -> dict[str, str]:
    """
    Constructs the item prices for the items from the price guide.

    Parameters
    ----------
    google_service_account_json : str
        The path to the Google Service Account JSON file.

    Returns
    -------
    item_prices : dict[str, str]
        A dictionary of item names and their prices.
    """
    item_prices = {}

    def transform_item_name(name: str) -> str:
        return (
            re.sub(r"\s+", " ", name.split("(")[0])
            .replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("'N'", "'n'")
            .strip()
        )

    def transform_item_price(price: str) -> str:
        return (
            ""
            if price.lower().strip()
            in ["", "?", "n/a", "n / a", "tbd", "t.b.d.", "free", "0"]
            else price.split("-")[-1].replace("+", "").strip()
        )

    print(f"Constructing item prices from {PRICE_GUIDE_URL}...")

    try:
        client = gspread.service_account(filename=google_service_account_json)
        sheet = client.open_by_url(PRICE_GUIDE_URL)
        sheets = sheet.worksheets()

        for sheet in sheets[1:-2]:
            data = sheet.get_all_values()

            headers = data[0]
            header_groups = []
            index_object = {"name": -1, "price": -1}

            for i, header in enumerate(headers):
                if header.strip().startswith("Name"):
                    index_object["name"] = i
                if header.strip().startswith("Price"):
                    index_object["price"] = i
                if index_object["name"] != -1 and index_object["price"] != -1:
                    header_groups.append(index_object)
                    index_object = {"name": -1, "price": -1}

            for row in data[1:]:
                for header_group in header_groups:
                    name = transform_item_name(row[header_group["name"]])
                    price = transform_item_price(row[header_group["price"]])
                    if price:
                        item_prices[name] = price

    except Exception as e:
        print(f"Error constructing item prices: {e}")

    return item_prices


def main() -> None:
    """
    Main function to prepare the embeddings for the icons and the ONNX model to get the embeddings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-embeddings-path",
        type=str,
        default="item-recognition-web/public/icon_embeddings",
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        default="item-recognition-web/public/embedder.onnx",
    )
    parser.add_argument(
        "--output-labels-path", type=str, default="item-recognition-web/src/labels"
    )
    parser.add_argument("--input-pack-zip", type=str, default="retrobution_r4.zip")
    parser.add_argument("--resource-dir", type=str, default="resources")
    parser.add_argument(
        "--google-service-account-json",
        type=str,
        default="resources/google-service-account.json",
    )
    args = parser.parse_args()

    pack_path = Path(args.input_pack_zip)
    resource_path = Path(args.resource_dir)
    output_embeddings_path = Path(args.output_embeddings_path)
    output_model_path = Path(args.output_model_path)
    output_labels_path = Path(args.output_labels_path)

    # download the pack zip
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_zip_path = pack_path
    pack_dir_path = (
        pack_path.parent / f"unpacked_{pack_path.relative_to(pack_path.parent).stem}"
    )

    if not pack_dir_path.is_dir():
        print(f"Downloading pack zip to {pack_zip_path}...")
        asyncio.run(
            download_file(f"{REPO_RELEASE_URL}/{args.input_pack_zip}", pack_zip_path)
        )

        # unzip the pack zip
        print(f"Unzipping pack zip to {pack_dir_path}...")
        shutil.unpack_archive(pack_zip_path, pack_dir_path)

        # remove the pack zip
        pack_zip_path.unlink()

    # copy icon assets to the output directory
    icon_source_path = pack_dir_path / "icons"
    icon_dest_path = output_embeddings_path.parent / "icons"
    if icon_dest_path.is_dir():
        shutil.rmtree(icon_dest_path)
    shutil.copytree(icon_source_path, icon_dest_path)

    # fetch item prices if possible
    item_prices = construct_item_prices(args.google_service_account_json)

    # construct the index and embedder
    construct_index_and_embedder(
        pack_dir_path,
        resource_path,
        output_embeddings_path,
        output_model_path,
        output_labels_path,
        item_prices,
    )


if __name__ == "__main__":
    main()
