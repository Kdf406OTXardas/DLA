import numpy as np
import pandas as pd
from pathlib import Path

SIZE_FIELD = 3
list_to_numpy = [x + 1 for x in range(SIZE_FIELD**3)]
input_field = np.array(list_to_numpy)
print(input_field)

df = pd.DataFrame(input_field, columns=['property'])
df['z'] = (df.index / (SIZE_FIELD * SIZE_FIELD)).astype(int)
df['y'] = (df.index / SIZE_FIELD).astype(int) - ((df.index / SIZE_FIELD / SIZE_FIELD) * SIZE_FIELD).astype(int)


# df['y'] = ((df.index / SIZE_FIELD) % SIZE_FIELD).astype(int)
df['x'] = (df.index - SIZE_FIELD * (df.index / SIZE_FIELD).astype(int)).astype(int)
print(df)

df.to_csv('/home/natkachov/datasets/export_from_dla_cuda/test.csv', index=False)
