# Item Recognition and Shopmaking Web

Retrobution shoplist website at https://retrobution-shoplist.pages.dev

## Setup

```sh
npm install
```

Please also go a directory above and run `src/prepare_embeds_model.py` once, as this will put the embeddings, labels and the model where they need to be.

## Local Development

```sh
npm run dev
```

## Deployment

```sh
npm run build && npx wrangler pages deploy dist/
```
