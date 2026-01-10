# Item Recognition and Shopmaking Web

Retrobution shoplist website at https://openfusion-crate-drop-analyzer.pages.dev

Latest used revision is: <code>beta-20111013_r4_academy</code>

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
npm run build && npx wrangler pages deploy dist/ --project-name openfusion-crate-drop-analyzer
```
