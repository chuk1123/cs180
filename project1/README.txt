CS180 Project 1

After extracting submission.zip, open a terminal in code/:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt

Create data/ beside main.py and place the 14 provided JPG/TIFF glass plates in it.
Use the original full-resolution TIFF scans, not resized previews. Then run:
  python main.py

The program runs NCC and L2 pyramid alignment on every scan, plus both
single-scale searches on the small JPGs. Results go into output/pyramid/ncc,
output/pyramid/l2, output/single_scale/ncc, and output/single_scale/l2.
Offsets are printed and saved in output/offsets.csv as (x, y), matching the
webpage. Functions internally return (dy, dx). Use Python 3.10 or newer.

For the three additional examples, download the original grayscale TIFF plates
from these Library of Congress records and name them as follows in data/:
  lugano.tif: https://www.loc.gov/pictures/item/2018679128/
  bashkir_woman_folk_costume.tif: https://www.loc.gov/pictures/item/2018679647/
  compound_locomotive.tif: https://www.loc.gov/pictures/item/2018679302/
No image files are included in this ZIP, as required by the assignment.

Run the optional Emir baseline, red-through-green, and gradient experiments with:
  python main.py --emir-experiments
These outputs go into output/emir_experiments. The webpage features the
red-through-green experiment. Gradient NCC is an optional separate experiment.

Project webpage:
  https://chuk1123.github.io/cs180/proj1/
