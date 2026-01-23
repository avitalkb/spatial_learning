# ============================================================================
# FILE 1: config.py
# ============================================================================

"""Configuration and constants for mobility attention project."""

# Data URLs
FOURSQUARE_URL = "https://raw.githubusercontent.com/nicolemalik/UPTDNet_dataset/main/Foursquare_Washington_Baltimore.csv"
GOWALLA_URL = "https://raw.githubusercontent.com/nicolemalik/UPTDNet_dataset/main/Gowalla_Dallas_Austin.csv"

# Model hyperparameters
EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30

# Data parameters
MIN_CATEGORY_COUNT = 100
MIN_TRAJECTORY_LENGTH = 5
MAX_TRAJECTORY_LENGTH = 15

# Category grouping keywords
CATEGORY_KEYWORDS = {
    "Food": ["restaurant", "food", "mexican", "american", "asian", "thai", "italian", 
             "chinese", "indian", "bbq", "pizza", "burger", "taco", "steak", "sushi",
             "seafood", "vietnamese", "greek", "diner", "sandwich", "breakfast",
             "chick-fil-a", "chipotle", "mcdonald", "wendy", "wings", "noodle"],
    
    "Coffee/Dessert": ["coffee", "starbucks", "dessert", "bakery", "ice cream", "candy",
                       "donut", "cafe", "tea", "juice", "smoothie", "yogurt"],
    
    "Bar/Nightlife": ["bar", "pub", "nightlife", "lounge", "brewery", "wine", "saloon",
                      "club", "nightclub", "beer", "cocktail", "tavern", "live music"],
    
    "Shopping": ["shop", "store", "mall", "grocery", "market", "retail", "apparel",
                 "clothing", "fashion", "furniture", "ikea", "target", "walmart"],
    
    "Work": ["office", "corporate", "work", "coworking", "warehouse", "industrial"],
    
    "Home": ["apartment", "home", "residential", "house", "condo"],
    
    "Transit": ["airport", "train", "bus", "station", "transit", "gas", "automotive",
                "parking", "car", "taxi", "uber", "metro", "rail"],
    
    "Entertainment": ["cinema", "theater", "theatre", "movie", "museum", "stadium",
                      "arena", "concert", "casino", "bowling", "arcade", "gallery"],
    
    "Outdoors": ["park", "garden", "plaza", "beach", "trail", "fountain", "monument",
                 "bridge", "lake", "river", "nature", "outdoor", "hiking"],
    
    "Hotel": ["hotel", "motel", "inn", "resort", "lodging", "hostel", "marriott"],
    
    "Services": ["salon", "barber", "gym", "fitness", "spa", "bank", "tattoo",
                 "laundry", "repair", "clinic", "hospital", "doctor", "pharmacy"],
    
    "Education": ["school", "university", "college", "library", "campus", "education"],
    
    "Zoo/Nature": ["zoo", "aquarium", "aviary", "reptile", "animal", "wildlife"],
    
    "Sports": ["sports", "field", "court", "golf", "tennis", "basketball", "baseball",
               "football", "soccer", "yoga", "crossfit"],
}
