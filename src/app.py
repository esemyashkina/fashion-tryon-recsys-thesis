"""Gradio interface for catalog recommendations and virtual try-on."""

import argparse
from functools import lru_cache
import os

import gradio as gr

from .catalog import FILTER_FIELDS, candidate_skus
from .evaluate import ROOT, load_split, read
from .recommendation import Terra, catalog_index, load_encoder, rank
from .tryon import CATEGORIES, PHOTO_TYPES, open_photo, try_on

FOLDER, _, PEOPLE, CATALOG = load_split("validation")
CONFIG = read(ROOT / "data/dataset.json")["recommendation"]
SELECTION_PROMPT = read(ROOT / "prompts/final_selection.json")[CONFIG["selection_prompt"]]


def open_product(sku):
    return open_photo(FOLDER / CATALOG[sku]["image_path"])


def person_photo(upload, person_id):
    if upload is not None:
        return upload
    if person_id in PEOPLE:
        return open_photo(FOLDER / PEOPLE[person_id]["image_path"])
    raise gr.Error("Загрузите фото человека или выберите пример из валидации.")


def sample_preview(person_id):
    return person_photo(None, person_id) if person_id else None


def product_options(category):
    return [(f"{item['garment_type']} · {item['color']} · {item['style']} · {number:02}", sku)
            for number, (sku, item) in enumerate(CATALOG.items(), 1) if item["category"] == category]


def filter_choices(category):
    return [gr.Dropdown(choices=sorted({item[field] for item in CATALOG.values()
                                       if item["category"] == category}), value=None, label=label)
            for field, label in zip(FILTER_FIELDS, ("Тип вещи", "Цвет", "Стиль"))]


@lru_cache(maxsize=1)
def recommendation_assets():
    encoder = load_encoder()
    vectors = catalog_index(encoder, FOLDER, CATALOG)
    return encoder, vectors


def run_recommendation(upload, person_id, category, garment_type, color, style):
    photo = person_photo(upload, person_id)
    filters = {"category": category}
    filters.update({field: value for field, value in zip(FILTER_FIELDS, (garment_type, color, style)) if value})
    pool = candidate_skus(filters)
    if not pool:
        raise gr.Error("В каталоге нет вещей с этой комбинацией фильтров.")
    from openai import OpenAIError

    try:
        terra = Terra(CONFIG["model"], CONFIG["reasoning_effort"])
        query, context = terra.query(photo, filters)
        encoder, vectors = recommendation_assets()
        shortlist = rank(encoder, vectors, query["query"], pool)[:CONFIG["k"]]
        pick = terra.choose(SELECTION_PROMPT, context, shortlist, open_product)
    except OpenAIError as exc:
        raise gr.Error("Не удалось обратиться к Terra. Проверьте OPENAI_API_KEY, доступ к модели и сеть.") from exc
    gallery = [(open_product(sku), f"{n}. {CATALOG[sku]['garment_type']} · {CATALOG[sku]['color']}")
               for n, sku in enumerate(shortlist, 1)]
    return open_product(pick["sku"]), query["query"], pick["reason"], gallery, pick["sku"], photo


def render_tryon(photo, garment, category, photo_type, steps):
    try:
        return try_on(photo, garment, category, photo_type, int(steps))
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc


def try_direct(upload, person_id, garment, sku, category, photo_type, steps):
    photo = person_photo(upload, person_id)
    if garment is None:
        if sku not in CATALOG or CATALOG[sku]["category"] != category:
            raise gr.Error("Загрузите фото вещи или выберите вещь из каталога.")
        garment = open_product(sku)
        photo_type = "flat-lay"
    return render_tryon(photo, garment, category, photo_type, steps)


def try_recommended(photo, sku, steps):
    if sku not in CATALOG or photo is None:
        raise gr.Error("Сначала подберите вещь.")
    return render_tryon(photo, open_product(sku), CATALOG[sku]["category"], "flat-lay", steps)


def build_app():
    people_options = [(f"Образ {n:02}", uid) for n, uid in enumerate(PEOPLE, 1)]
    with gr.Blocks(title="Виртуальная примерка и подбор одежды", analytics_enabled=False) as demo:
        gr.Markdown("# Виртуальная примерка и подбор одежды")
        gr.Markdown("Загрузите фото человека или выберите готовый образ. При загрузке используется ваше фото.")
        with gr.Tab("Подбор из каталога"):
            with gr.Row():
                person_rec = gr.Image(label="Фото человека", type="pil")
                with gr.Column():
                    id_rec = gr.Dropdown(people_options, label="Пример из валидации")
                    id_rec.change(sample_preview, id_rec, person_rec)
                    cat_rec = gr.Dropdown(CATEGORIES, value="tops", label="Категория")
                    initial = filter_choices("tops")
                    cat_rec.change(filter_choices, cat_rec, initial)
                    recommend = gr.Button("Подобрать вещь", variant="primary")
            with gr.Row():
                chosen = gr.Image(label="Выбранная вещь", interactive=False)
                result = gr.Image(label="Результат примерки", interactive=False)
            with gr.Row():
                steps_rec = gr.Slider(1, 50, value=10, step=1, label="Шаги примерки")
                fit = gr.Button("Примерить выбранную вещь")
            query = gr.Textbox(label="Поисковый запрос", interactive=False)
            reason = gr.Textbox(label="Объяснение выбора", interactive=False)
            gallery = gr.Gallery(label="Кандидаты в порядке FashionCLIP", columns=6)
            sku_state = gr.State()
            photo_state = gr.State()
            recommend.click(
                run_recommendation, [person_rec, id_rec, cat_rec, *initial],
                [chosen, query, reason, gallery, sku_state, photo_state],
            )
            fit.click(try_recommended, [photo_state, sku_state, steps_rec], result, concurrency_id="tryon")
        with gr.Tab("Примерка выбранной вещи"):
            with gr.Row():
                person_own = gr.Image(label="Фото человека", type="pil")
                garment_own = gr.Image(label="Фото своей вещи", type="pil")
                with gr.Column():
                    id_own = gr.Dropdown(people_options, label="Пример из валидации")
                    id_own.change(sample_preview, id_own, person_own)
                    cat_own = gr.Dropdown(CATEGORIES, value="tops", label="Категория")
                    sku_own = gr.Dropdown(product_options("tops"), label="Вещь из каталога")
                    cat_own.change(lambda c: gr.Dropdown(choices=product_options(c), value=None), cat_own, sku_own)
                    photo_type = gr.Dropdown(PHOTO_TYPES, value="flat-lay", label="Тип фото своей вещи")
                    steps_own = gr.Slider(1, 50, value=10, step=1, label="Шаги примерки")
            run = gr.Button("Примерить", variant="primary")
            result_own = gr.Image(label="Результат примерки", interactive=False)
            run.click(try_direct, [person_own, id_own, garment_own, sku_own, cat_own, photo_type, steps_own],
                      result_own, concurrency_id="tryon")
    return demo.queue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    username, password = os.getenv("GRADIO_USERNAME"), os.getenv("GRADIO_PASSWORD")
    if bool(username) != bool(password):
        parser.error("Set both GRADIO_USERNAME and GRADIO_PASSWORD.")
    build_app().launch(server_name=args.host, server_port=args.port, share=False,
                       auth=(username, password) if username else None)


if __name__ == "__main__":
    main()
