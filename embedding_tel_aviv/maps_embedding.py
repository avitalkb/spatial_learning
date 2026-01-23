# urban_embeddings.py
# Clean, organized version of your urbanism embedding code
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_filter_data(csv_path='tel_aviv_pois_clean.csv', min_count=50):
    """Load POI data and filter to categories with enough examples"""
    df = pd.read_csv(csv_path)
    
    # Keep only categories with min_count examples
    top_categories = df['category'].value_counts()
    keep_categories = top_categories[top_categories >= min_count].index.tolist()
    df_filtered = df[df['category'].isin(keep_categories)].copy()
    
    print(f"Filtered from {len(df)} to {len(df_filtered)} POIs")
    print(f"Categories: {df_filtered['category'].nunique()}")
    
    return df_filtered


# ============================================================================
# 2. SPATIAL CONTEXT CREATION
# ============================================================================

def build_spatial_tree(df):
    """Build KD-tree for fast spatial queries"""
    coords = df[['lat', 'lon']].values
    tree = cKDTree(coords)
    return tree, coords


def create_spatial_sentences(df, tree, k_neighbors=10):
    """Create 'sentences' where each POI is surrounded by its neighbors"""
    coords = df[['lat', 'lon']].values
    distances, indices = tree.query(coords, k=k_neighbors+1)
    
    sentences = []
    for poi_idx in range(len(df)):
        center_category = df.iloc[poi_idx]['category']
        neighbor_indices = indices[poi_idx][1:11]  # Skip self
        neighbor_categories = df.iloc[neighbor_indices]['category'].tolist()
        sentence = [center_category] + neighbor_categories
        sentences.append(sentence)
    
    return sentences, indices


# ============================================================================
# 3. EMBEDDING TRAINING
# ============================================================================

def train_embeddings(sentences, vector_size=50, window=10, epochs=20):
    """Train Word2Vec on spatial sentences"""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        epochs=epochs
    )
    
    print(f"Trained embeddings for {len(model.wv)} categories")
    return model


# ============================================================================
# 4. AREA CHARACTER COMPUTATION
# ============================================================================

def compute_food_counts(df, tree, radius_km=0.3, n_grid=75, min_pois=5):
    """Compute simple food POI counts on grid"""
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    lats = np.linspace(lat_min, lat_max, n_grid)
    lons = np.linspace(lon_min, lon_max, n_grid)
    
    radius_deg = radius_km / 111
    
    food_cats = ['restaurant', 'cafe', 'bar', 'fast_food', 'pub']
    
    food_counts = []
    grid_lats = []
    grid_lons = []
    
    for lat in lats:
        for lon in lons:
            nearby_indices = tree.query_ball_point([lat, lon], radius_deg)
            
            if len(nearby_indices) >= min_pois:
                count = df.iloc[nearby_indices]['category'].isin(food_cats).sum()
                food_counts.append(count)
                grid_lats.append(lat)
                grid_lons.append(lon)
    
    # Normalize
    scaler = StandardScaler()
    food_counts_norm = scaler.fit_transform(np.array(food_counts).reshape(-1, 1)).flatten()
    
    return grid_lats, grid_lons, food_counts_norm


def compute_area_embeddings(df, tree, model, radius_km=0.3, n_grid=75, min_pois=5):
    """Compute embedding-based 'character' for grid cells"""
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    lats = np.linspace(lat_min, lat_max, n_grid)
    lons = np.linspace(lon_min, lon_max, n_grid)
    
    radius_deg = radius_km / 111
    
    area_embeddings = []
    grid_lats = []
    grid_lons = []
    
    for lat in lats:
        for lon in lons:
            nearby_indices = tree.query_ball_point([lat, lon], radius_deg)
            
            if len(nearby_indices) >= min_pois:
                nearby_categories = df.iloc[nearby_indices]['category']
                nearby_embeds = [model.wv[cat] for cat in nearby_categories]
                area_embed = np.mean(nearby_embeds, axis=0)
                
                area_embeddings.append(area_embed)
                grid_lats.append(lat)
                grid_lons.append(lon)
    
    area_embeddings = np.array(area_embeddings)
    
    # Reduce to 1D for visualization
    pca = PCA(n_components=1)
    area_scores = pca.fit_transform(area_embeddings).flatten()
    
    # Standardize
    scaler = StandardScaler()
    area_scores_norm = scaler.fit_transform(area_scores.reshape(-1, 1)).flatten()
    
    return grid_lats, grid_lons, area_scores_norm, area_embeddings


# ============================================================================
# 5. STREET NETWORK LOADING
# ============================================================================

def load_street_network():
    """Load Tel Aviv street network using OSMnx"""
    try:
        import osmnx as ox
        
        print("Downloading street network...")
        G = ox.graph_from_place(
            'Tel Aviv, Israel', 
            network_type='drive',
            custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary"]'
        )
        
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        print(f"Downloaded {len(edges)} street segments")
        return edges
    
    except ImportError:
        print("Warning: osmnx not installed. Install with: pip install osmnx")
        return None
    except Exception as e:
        print(f"Warning: Could not download streets: {e}")
        return None


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def plot_comparison(food_grid, embedding_grid, df, streets=None):
    """
    Plot side-by-side comparison of food counts vs embeddings
    
    Parameters:
    -----------
    food_grid : tuple of (lats, lons, scores)
    embedding_grid : tuple of (lats, lons, scores)
    df : DataFrame with POI data
    streets : GeoDataFrame with street network (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    food_lats, food_lons, food_scores = food_grid
    emb_lats, emb_lons, emb_scores = embedding_grid
    
    vmin, vmax = -2, 2  # Consistent scale
    
    # Left panel: Food counts
    if streets is not None:
        streets.plot(ax=axes[0], linewidth=0.5, edgecolor='gray', alpha=0.5, zorder=1)
    
    im1 = axes[0].scatter(food_lons, food_lats, c=food_scores, 
                          s=10, alpha=0.6, cmap='RdYlBu', vmin=vmin, vmax=vmax)
    axes[0].set_title('Simple Counting:\nFood Places', fontsize=14, weight='bold')
    plt.colorbar(im1, ax=axes[0], label='Food density (normalized)')
    
    # Right panel: Embeddings
    if streets is not None:
        streets.plot(ax=axes[1], linewidth=0.5, edgecolor='gray', alpha=0.5, zorder=1)
    
    im2 = axes[1].scatter(emb_lons, emb_lats, c=emb_scores, 
                          s=10, alpha=0.6, cmap='RdYlBu', vmin=vmin, vmax=vmax)
    axes[1].set_title('Embeddings:\nUrban Character', fontsize=14, weight='bold')
    plt.colorbar(im2, ax=axes[1], label='Urban character (normalized)')
    
    # Set consistent limits
    for ax in axes:
        ax.set_xlim(df['lon'].min() - 0.01, df['lon'].max() + 0.01)
        ax.set_ylim(df['lat'].min() - 0.01, df['lat'].max() + 0.01)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.show()


def plot_urban_character(grid_lons, grid_lats, area_scores, df, streets=None, 
                        title='Urban Character'):
    """Plot single urban character heatmap with streets"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Streets first (background)
    if streets is not None:
        streets.plot(ax=ax, linewidth=0.5, edgecolor='gray', alpha=0.5, zorder=1)
    
    # Heatmap on top
    scatter = ax.scatter(grid_lons, grid_lats, c=area_scores, 
                        s=20, alpha=0.7, cmap='RdYlBu', vmin=-2, vmax=2, zorder=10)
    plt.colorbar(scatter, label='Urban Character', ax=ax)
    
    ax.set_xlim(df['lon'].min() - 0.01, df['lon'].max() + 0.01)
    ax.set_ylim(df['lat'].min() - 0.01, df['lat'].max() + 0.01)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

class UrbanEmbeddings:
    """Main class to encapsulate the entire pipeline"""
    
    def __init__(self, csv_path='tel_aviv_pois_clean.csv'):
        self.df = load_and_filter_data(csv_path)
        self.tree, self.coords = build_spatial_tree(self.df)
        self.sentences = None
        self.indices = None
        self.model = None
        self.grid_data = {}
        self.streets = None
        
    def train(self):
        """Train the embedding model"""
        self.sentences, self.indices = create_spatial_sentences(self.df, self.tree)
        self.model = train_embeddings(self.sentences)
        return self.model
    
    def load_streets(self):
        """Load street network"""
        self.streets = load_street_network()
        return self.streets
    
    def compute_grids(self, radius_km=0.3, n_grid=75):
        """Compute both food counts and embedding grids"""
        # Food counts
        food_lats, food_lons, food_scores = compute_food_counts(
            self.df, self.tree, radius_km, n_grid
        )
        
        # Embeddings
        emb_lats, emb_lons, emb_scores, emb_embeddings = compute_area_embeddings(
            self.df, self.tree, self.model, radius_km, n_grid
        )
        
        self.grid_data = {
            'food': (food_lats, food_lons, food_scores),
            'embedding': (emb_lats, emb_lons, emb_scores),
            'embeddings_full': emb_embeddings
        }
        
        return self.grid_data
    
    def plot_comparison(self):
        """Plot side-by-side comparison"""
        if not self.grid_data:
            raise ValueError("Must run compute_grids() first")
        
        plot_comparison(
            self.grid_data['food'],
            self.grid_data['embedding'],
            self.df,
            self.streets
        )
    
    def plot_embeddings(self):
        """Plot just the embedding-based character map"""
        if not self.grid_data:
            raise ValueError("Must run compute_grids() first")
        
        emb_lats, emb_lons, emb_scores = self.grid_data['embedding']
        plot_urban_character(emb_lons, emb_lats, emb_scores, self.df, 
                           self.streets, 'Urban Character')
    
    def find_similar_categories(self, category, topn=5):
        """Find categories similar to the given one"""
        if self.model is None:
            raise ValueError("Must run train() first")
        return self.model.wv.most_similar(category, topn=topn)
    
    def describe_area(self, lat, lon, radius_km=0.3):
        """Describe what POIs are in a given area"""
        radius_deg = radius_km / 111
        nearby_indices = self.tree.query_ball_point([lat, lon], radius_deg)
        
        if len(nearby_indices) > 0:
            nearby_cats = self.df.iloc[nearby_indices]['category'].value_counts()
            return nearby_cats
        return None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
# %%
if __name__ == "__main__":
    # Initialize
    urban = UrbanEmbeddings('tel_aviv_pois_filtered.csv')
    
    # Train embeddings
    print("Training embeddings...")
    urban.train()
    
    # Load street network (optional but recommended)
    print("Loading street network...")
    urban.load_streets()
    
    # Compute both grids
    print("Computing grids...")
    urban.compute_grids(radius_km=0.3, n_grid=75)
    
    # Plot comparison (THIS IS YOUR LATEST FIGURE!)
    print("Plotting comparison...")
    urban.plot_comparison()
    
    # Optional: plot just embeddings
    # urban.plot_embeddings()
    
    # Query examples
    print("\nSimilar to 'restaurant':")
    print(urban.find_similar_categories('restaurant'))
    
    print("\nWhat's near Dizengoff Center (32.0796, 34.7753)?")
    print(urban.describe_area(32.0796, 34.7753, radius_km=0.3))
# %%
