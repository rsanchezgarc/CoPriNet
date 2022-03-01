import os


TEST_SIZE= int(2e5)
CSV_CHUNK_SIZE= int(1e5)

RAW_DATA_DIRNAME= "data/MCule" #Can be provided via command line arguments
BUILDING_BLOCKS_FNAME= os.path.join(RAW_DATA_DIRNAME, "mcule_purchasable_building_blocks_210806.smi.gz")

DATASET_DIRNAME = "data/mcule_chunks"

ENCODED_DIR = os.path.join(DATASET_DIRNAME, "encoded")

RAW_DATA_FILE_SUFFIX = r"mcule_full_(train|test|val)\.csv_split_\w+\.csv$"

NUM_BUCKETS = 100
N_CPUS = 12

PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS = 25

MULTIPROC_PREFETCH_FACTOR = 4
MULTIPROC_TIMEOUT = 60
PERSISTENT_WORKERS = False

BATCH_SIZE = 512
NUM_WORKERS_PER_GPU = 2

USE_MMOL_INSTEAD_GRAM = True

RANDOM_SEED = 121
LEARNING_RATE= 1e-5

N_LAYERS=6
N_HIDDEN_NODE=75
N_HIDDEN_EDGE=50

DATA_DIR = os.path.abspath(os.path.join(__file__, "../.."))
DEFAULT_MODEL = os.path.join( DATA_DIR, "data/models/final_inStock/lightning_logs/version_0/checkpoints/epoch=238-step=1295857.ckpt")
# DEFAULT_MODEL = os.path.join( DATA_DIR, "data/models/final_virtual/lightning_logs/version_0/checkpoints/epoch=172-step=3231639.ckpt")