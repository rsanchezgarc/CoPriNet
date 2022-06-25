import os

################## Common options ##################
NUM_BUCKETS = 100
N_CPUS = 12

MULTIPROC_PREFETCH_FACTOR = 4
MULTIPROC_TIMEOUT = 60
PERSISTENT_WORKERS = False

BATCH_SIZE = 512
NUM_WORKERS_PER_GPU = 2
BUFFER_N_BATCHES_FOR_PRED=50 #Process as many batches together for speed. Set it as large as possible to fit your RAM memory

USE_MMOL_INSTEAD_GRAM = True

RANDOM_SEED = 121

################## Net  ##################
N_LAYERS=10
N_HIDDEN_NODE=75
N_HIDDEN_EDGE=50

LEARNING_RATE= 1e-5
PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS = 25
COSINE_LR_SCHEDULE_N_EPOCHS = 10

############# Data ##################

TEST_SIZE= int(2e5)
CSV_CHUNK_SIZE= int(1e5)

DATA_DIR = os.path.abspath(os.path.join(__file__, "../.."))

RAW_DATA_DIRNAME= "data/MCule" #Can be provided via command line arguments
BUILDING_BLOCKS_FNAME= os.path.join(RAW_DATA_DIRNAME, "mcule_purchasable_building_blocks_210806.smi.gz")

DATASET_DIRNAME = "data/mcule_chunks"
DATASET_IN_MEMORY = True # Set to true to load the dataset into memory
ENCODED_DIR = os.path.join(DATASET_DIRNAME, "encoded")

RAW_DATA_FILE_SUFFIX = r"mcule_full_(train|test|val)\.csv_split_\w+\.csv$"

DEFAULT_MODEL = os.path.join( DATA_DIR, "data/models/final_inStock/lightning_logs/version_0/checkpoints/epoch=238-step=1295857.ckpt")
# DEFAULT_MODEL = os.path.join( DATA_DIR, "data/models/final_virtual/lightning_logs/version_0/checkpoints/epoch=172-step=3231639.ckpt")


######################### Ignore the next lines if not running a comparison ################
USE_FEATURES_NET = False       #set this to True to use a MLP instead of GNN
NORMALIZE_NODE_FEATS = False   #set this to True if USE_FEATURES_NET==True