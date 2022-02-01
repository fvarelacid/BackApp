### Generation of more data ###

# ------------------------------------------
# 1. Load data
# 2. Preprocess data until we have 12s audios
# 3. Loop through the tensor
# 3.1. Apply time_shift x 3 to the tensor
# 3.2. Convert tensors to mel spectrograms
# 3.3. Apply mel_augment x 3 to the tensor
# 3.4. Return tensors
# ------------------------------------------

import preprocessing
import torchaudio

