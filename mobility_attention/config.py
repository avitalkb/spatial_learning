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
HOUR_EMBED_DIM = 8
DOW_EMBED_DIM = 4
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
             "chick-fil-a", "chipotle", "mcdonald", "wendy", "wings", "noodle",
             "fine dining", "mediterranean", "french", "vegetarian", "vegan",
             "hot dog", "sausage", "potbelly", "sonic", "dairy queen",
             "krispy kreme", "mellow mushroom", "fish & chips", "african",
             "middle eastern", "street fare", "snow cone", "doughnuts"],

    "Coffee/Dessert": ["coffee", "starbucks", "dessert", "bakery", "ice cream", "candy",
                       "donut", "cafe", "tea", "juice", "smoothie", "yogurt",
                       "chocolate"],

    "Bar/Nightlife": ["bar", "pub", "nightlife", "lounge", "brewery", "wine", "saloon",
                      "club", "nightclub", "beer", "cocktail", "tavern", "live music"],

    "Shopping": ["shop", "store", "mall", "grocery", "market", "retail", "apparel",
                 "clothing", "fashion", "furniture", "ikea", "target", "walmart",
                 "shoes", "walgreens", "best buy", "lowe", "costco", "nordstrom",
                 "rei", "gap", "old navy", "urban outfitters", "victoria's secret",
                 "paper goods", "accessories", "antiques", "toys", "at&t",
                 "verizon", "sprint", "t-mobile", "ups", "design within reach",
                 "tobacco"],

    "Work": ["office", "corporate", "work", "coworking", "warehouse", "industrial",
             "administration", "government", "city hall", "capitol"],

    "Home": ["apartment", "home", "residential", "house", "condo", "duplex"],

    "Transit": ["airport", "train", "bus", "station", "transit", "gas", "automotive",
                "parking", "car", "taxi", "uber", "metro", "rail",
                "subway", "terminal", "gate"],

    "Entertainment": ["cinema", "theater", "theatre", "movie", "museum", "stadium",
                      "arena", "concert", "casino", "bowling", "arcade", "gallery",
                      "cineplex", "convention center", "performing arts", "dancefloor",
                      "music", "karaoke", "film", "exhibit", "planetarium",
                      "party", "special event", "meet up", "conference",
                      "sxsw", "birthday", "wedding", "watch party", "holiday",
                      "interactive", "screening", "showcase",
                      "other - entertainment"],

    "Outdoors": ["park", "garden", "plaza", "beach", "trail", "fountain", "monument",
                 "bridge", "lake", "river", "nature", "outdoor", "hiking",
                 "playground", "scenic lookout", "farm", "campground", "cave",
                 "canal", "waterway", "vineyard", "glade", "big blue wet thing"],

    "Hotel": ["hotel", "motel", "inn", "resort", "lodging", "hostel", "marriott",
              "four seasons"],

    "Services": ["salon", "barber", "gym", "fitness", "spa", "bank", "tattoo",
                 "laundry", "repair", "clinic", "hospital", "doctor", "pharmacy",
                 "other - services", "other - medical", "medical", "dentist",
                 "veterinarian"],

    "Education": ["school", "university", "college", "library", "campus", "education",
                  "student center", "lab", "dormitory"],

    "Zoo/Nature": ["zoo", "aquarium", "aviary", "reptile", "animal", "wildlife",
                   "primates", "tigers", "lions", "elephants", "giraffes",
                   "zebras", "aquatics"],

    "Sports": ["sports", "field", "court", "golf", "tennis", "basketball", "baseball",
               "football", "soccer", "yoga", "crossfit",
               "running", "racetrack", "rec center", "ice skating"],

    "Culture": ["historic", "landmark", "architecture", "sculpture", "castle",
                "victorian", "skyscraper", "tower", "hall", "pavilion",
                "arts", "art & culture", "crafts", "creative",
                "craftsman", "modern"],

    "Religious": ["church", "temple", "mission", "place of worship", "cemetery"],

    "Tech": ["technology"],
}
