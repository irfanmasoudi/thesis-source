import pathlib

PROJECT_DIR = pathlib.Path(__file__).parent.parent

MODEL_DIR = pathlib.Path("/media/banana/data/models")
SAMPLE_DIR = pathlib.Path("/media/banana/data/samples")
RESULT_DIR = pathlib.Path("/media/banana/data/results")

DATA_DIR = pathlib.Path("/media/banana/data/dataset")
UDACITY_DATA_DIR = DATA_DIR.joinpath("deepnurse")
CAMELYON_DATA_DIR = DATA_DIR.joinpath("camelyon")
UDACITY_STYLE_DATA_DIR = DATA_DIR.joinpath("deepnurse_style_transfer")
FVIS_DATA_DIR = DATA_DIR.joinpath("fractals_and_fvis").joinpath("first_layers_resized256_onevis").joinpath("images")
FRACTALS_DATA_DIR = DATA_DIR.joinpath("fractals_and_fvis").joinpath("fractals").joinpath("images")
COCO_DATA_DIR = DATA_DIR.joinpath("coco2017")
WIKIART_DATA_DIR = DATA_DIR.joinpath("wikiart")