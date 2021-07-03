from pathlib import Path
from random import random
from cosa.transform import k_representative_pallette, elastic_transform
from PIL import Image
import numpy as np

FUNCTIONS = {
    "k_rep": k_representative_pallette,
    "elastic": elastic_transform,
}


class Cosa:
    """Main `cosa` class that has the functionality to
    read, transform and save images.
    """

    def __init__(self):
        self.image = None
        self.transformed = None

    def read(self, input_path: str):
        """Load an image, given a path."""
        input_path = Path(input_path)
        if input_path.is_file():
            self.image = np.asarray(Image.open(input_path))
        else:
            raise ValueError(f"File {input_path} does not exist!")

    def transform(self, func=None, **args):
        """Transform a loaded image with a given
        functions. If the function is not specified
        one will be chosen randomly.
        """
        funcs_list = list(FUNCTIONS.keys())
        if func not in funcs_list:
            raise ValueError(f"{func} function not supported.")
        if not func:
            func = random.choice(funcs_list)
        self.transformed = FUNCTIONS[func](img=self.img, **args)

    def save(self, output_path: str):
        """Save an image in a given path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.transformed is None:
            raise ValueError(
                "There is no image to save! Make sure to load & transform one first."
            )
        with output_path.open("wb") as file:
            file.write(self.transformed.content)
            file.close()
