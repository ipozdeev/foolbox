# econometrics toolbox!
# yeeeeeeay!
import pandas as pd

options = {
        'display': {
            'max_columns': 7,
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 21,
            'max_seq_items': 50,         # Max length of printed sequence
            'precision': 4,
            'show_dimensions': True
        }
    }

for category, option in options.items():
    for op, value in option.items():
        pd.set_option(".".join((category, op)), value)
