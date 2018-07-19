"""

Trainer module controls:
    * Data pre-processing
    * Iterations and quality of training
    * Tracks and provide training status

Data pre-processing: This is an important step of
any model training procedure, the data is searched,retrived
and saved in a format which is efficient to use later.
This procedure controls the complexity of data and evaluation

Quality of training: No one knows when to stop a training
procedure unless experianced enough to determine the tradeoff
between time and last ounce of accuracy. Hence, its important
to keep a check on the requirements when such training is
done in an automated way. This procedure provide options
like iterations, and time to train on a pre-processed data.

Tracking training status: This provides full stats once the
training is complete or interrupted in an automatic manner.

This module is GPLv3 licensed.
"""
