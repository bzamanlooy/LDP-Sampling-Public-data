
import warnings
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np
import gurobipy as gp
from scipy.optimize import brentq
import gc
import pandas as pd

warnings.filterwarnings('ignore')

def func(x, P, Q0, epsilon):
    return np.sum(np.minimum(np.maximum(Q0 / np.exp(epsilon / 2), x * P), np.exp(epsilon / 2) * Q0)) - 1

def find_root(P, Q0, epsilon, x_low, x_high):
    root = brentq(lambda x: func(x, P, Q0, epsilon), x_low, x_high)
    return root

def find_distribution(P,Q0,epsilon, C):
    Phat = []
    for i in range(len(P)):
      Phat.append(np.minimum(np.maximum(Q0[i] / np.exp(epsilon / 2), C * P[i]), np.exp(epsilon / 2) * Q0[i]))
    return Phat

def K_with_prior(q, eps):

  n = len(q)
  # print(n)
  if n==2:
    denominator = (np.exp(eps)*q[0] + q[1])
    K_2 = np.zeros((2,2))
    K_2[0,0] = np.exp(eps)*q[0]/denominator
    K_2[0,1] = 1 - K_2[0,0]
    K_2[1,0] = q[0]/denominator
    K_2[1,1] = 1 - K_2[1,0]
    return K_2
  else:
    denominator = (np.exp(eps)*q[0] + 1 - q[0])

    K = np.zeros((n,n))

    K[0,0] = np.exp(eps)*q[0]/denominator
    K[0,1:n] = q[1:]/denominator
    K[1:n, 0] = q[0]/denominator

    K_n_minus_1 = K_with_prior(q[1:]/np.sum(q[1:]), eps)

    a = 1 - q[0]/denominator
    K[1:(n), 1:(n)] = K_n_minus_1*a
    return K

def projected_distribution_with_prior_husain(p,q, eps):

  gp.setParam('OutputFlag', 0)
  eeps = float(np.exp(eps/2))
  m = gp.Model("Projected_distribution_with_prior")

  n = len(p)
  # define variables for the projected distribution
  l = m.addVars(n, lb = 0, ub =1, vtype = gp.GRB.CONTINUOUS, name = 'projected_distribution')
  # Deffine auxilary variables to mimic max(x,0) in the definition of TV distance
  Z = m.addVars(n, vtype = gp.GRB.CONTINUOUS, name = 'auxilary')

  # sum of convex multipliers has to be one
  m.addConstr(l.sum() == 1)

  # constaint for l to be in an epsilon mullifier of q
  for i in range(n):
    m.addConstr(l[i] <= q[i]*eeps)
    m.addConstr(l[i]*eeps >= q[i])

#   model.addConstr(x + y >= 1, "lower_bound")
# model.addConstr(x + y <= 5, "upper_bound")
  # Auxilary constraints
  for i in range(n):
    m.addConstr(Z[i] >= 0)
    m.addConstr(Z[i] >= p[i] - l[i])


  m.setObjective(Z.sum(), gp.GRB.MINIMIZE)
  m.optimize()

  phat = []
  for i in range(n):
    phat.append(l[i].x)
  return phat

def tv(p,q):
  return 0.5*np.sum(np.abs(p-q))

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def perform_projection_min_max(p,q , eps, K, f_div):
  K = K_with_prior(q,eps)
  phat_min_max = p@K
  if f_div == 'tv':
    return tv(p, phat_min_max)
  elif f_div == 'kl':
    return kl_divergence(p, phat_min_max)

def perform_projection_husain(p,q , eps, K, f_div):
  if f_div == 'tv':
    phat_husain = projected_distribution_with_prior_husain(p,q, eps)

  elif f_div == 'kl':
      x_low = 0
      x_high = 10000

      C = find_root(p, q, eps, x_low, x_high)
      phat_husain = find_distribution(p, q, eps, C)

  if f_div == 'tv':
    return tv(p, phat_husain)
  elif f_div == 'kl':
    return kl_divergence(p, phat_husain)


def perform_projection(p,q , eps, K, f_div):
  # K = K_with_prior(q,eps)
  phat_min_max = p@K

  if f_div == 'tv':
    phat_husain = projected_distribution_with_prior_husain(p,q, eps)

  elif f_div == 'kl':
      x_low = 0
      x_high = 1000

      C = find_root(p, q, eps, x_low, x_high)
      phat_husain = find_distribution(p, q, eps, C)

  if f_div == 'tv':
    return tv(p, phat_min_max), tv(p, phat_husain)
  elif f_div == 'kl':
    return kl_divergence(p, phat_min_max), kl_divergence(p, phat_husain)




def process_click_data(df, column_name, row_col, site_col='site_id', ntop=100):
    """
    Process the click data to compute distributions and their average based on a specified row column and site column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing click data.
    - row_col (str): The column name to use for rows in the pivot table (e.g., 'device_id').
    - site_col (str): The column name for the site information (e.g., 'site_id').
    - ntop (int): Number of top site IDs to consider based on click counts.

    Returns:
    - distributions (list of lists): List of distributions for each unique value in row_col.
    - avg_distribution (list): Average distribution across all unique values in row_col.
    - pivot_table (pd.DataFrame): Pivot table of the distributions.
    """
    avg_distribution_list = []
    individual_distributions_list = []

    # Get sorted unique values for the specified column
    unique_values = np.sort(df[column_name].unique())


    threshold = 100

    category_list = []

    row_counts = df[row_col].value_counts()
    valid_row_cols = row_counts[row_counts >= 20].index
    df = df[df[row_col].isin(valid_row_cols)]

            # Ensure the necessary columns are correctly named
    df = df.rename(columns={site_col: 'site', 'click': 'click'})




    # Wrap the iteration with tqdm to show progress
    for value in tqdm(unique_values):
        df_cur = df[df[column_name] == value]



          # Find the top ntop most clicked site IDs in df_cur
        top_site_ids = df_cur.groupby('site')['click'].sum().nlargest(ntop).index


        # Filter the dataset to include only the top ntop most clicked site IDs
        top_sites_df = df_cur[df_cur['site'].isin(top_site_ids)]



        # Count clicks per row_col for these top site IDs
        row_clicks = top_sites_df.groupby([row_col, 'site'])['click'].count().reset_index(name='count')

        # Create a pivot table with row_col as rows and site as columns
        pivot_table = row_clicks.pivot_table(index=row_col, columns='site', values='count', fill_value=0)

        # Rename the columns to make them clear
        pivot_table.columns = [f'site_{category}' for category in pivot_table.columns]

        # Normalize the counts to sum to 1 for each row_col
        pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

        # Reset index to make row_col a column instead of an index
        pivot_table.reset_index(inplace=True)

        if len(pivot_table) < threshold:
          continue  # Skip this value if it does not meet the threshold
        else:
            category_list.append(value)

        # Convert the pivot table to a list of rows (excluding row_col column if necessary)
        selected_data = pivot_table.iloc[:, 1:]  # Exclude the first column (row_col) if necessary
        distributions = selected_data.values.tolist()

        # Compute the average distribution
        arr = np.array(distributions)
        avg_distribution = np.mean(arr, axis=0).tolist()

        avg_distribution_list.append(avg_distribution)
        individual_distributions_list.append(distributions)

        # Delete unnecessary variables at the end of each loop iteration
        del df_cur, top_site_ids, top_sites_df, row_clicks, pivot_table, selected_data, arr
        gc.collect()  # Force garbage collection

    return category_list, individual_distributions_list, avg_distribution_list



import os
import pandas as pd

def get_common_genres(input_dir_100k):
    """
    Get the intersection of genres for MovieLens 1M and 100K datasets.

    Args:
        input_dir_100k (str): Directory containing the MovieLens 100K dataset.

    Returns:
        list: List of common genres across both datasets.
    """
    # Construct file path using os.path.join
    genre_path_100k = os.path.join(input_dir_100k, 'u.genre')

    # Load genre mapping for MovieLens 100K
    genre_mapping_100k = pd.read_csv(genre_path_100k, sep='|', header=None, names=['Genre', 'Index']).dropna()
    genres_100k = set(genre_mapping_100k['Genre'])

    # Defined genre set for MovieLens 1M (since it's predefined in the dataset)
    genres_1m = {
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    }

    # Get intersection of genres
    common_genres = sorted(genres_100k & genres_1m)

    return common_genres


def get_individual_distributions_1m(common_genres, input_dir):
    """
    Process MovieLens 1M dataset to calculate individual distributions per user per age group.

    Args:
        common_genres (list): List of common genres.
        input_dir (str): Directory containing the dataset.

    Returns:
        dict: Individual distributions per user per age group.
    """
    # Construct file paths using os.path.join for cross-platform compatibility
    users_path = os.path.join(input_dir, 'users.dat')
    movies_path = os.path.join(input_dir, 'movies.dat')
    ratings_path = os.path.join(input_dir, 'ratings.dat')

    # Load datasets
    users = pd.read_csv(users_path, sep='::', header=None, engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    movies = pd.read_csv(movies_path, sep='::', header=None, encoding='latin-1', engine='python',
                         names=['MovieID', 'Title', 'Genres'])
    ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    # Create genre columns
    for genre in common_genres:
        movies[genre] = movies['Genres'].apply(lambda x: 1 if genre in x else 0)

    # Merge datasets
    data = pd.merge(pd.merge(ratings, users, on='UserID'), movies, on='MovieID')

    # Assign age groups
    data['Age Group'] = data['Age'].apply(
        lambda age: '18-24' if age <= 24 else
                    '25-34' if age <= 34 else
                    '35-44' if age <= 44 else
                    '45-49' if age <= 49 else
                    '50-54' if age <= 54 else
                    '55+'
    )

    # Display age group counts
    print("Age Group Counts:\n", data['Age Group'].value_counts())

    # Calculate sum of genre ratings per user per age group
    user_genre_ratings = data.groupby(['UserID', 'Age Group'])[common_genres].sum().reset_index()

    # Normalize genre ratings per user
    user_genre_ratings[common_genres] = user_genre_ratings[common_genres].div(user_genre_ratings[common_genres].sum(axis=1), axis=0).fillna(0)

    # Extract individual distributions
    individual_distributions = {
        age_group: group[common_genres].to_numpy().tolist()
        for age_group, group in user_genre_ratings.groupby('Age Group')
    }

    return individual_distributions


def get_avg_distributions_100k(common_genres, input_dir):
    """
    Process MovieLens 100K dataset to calculate average distributions per age group.

    Args:
        common_genres (list): List of common genres.
        input_dir (str): Directory containing the dataset.

    Returns:
        dict: Average distributions per age group.
    """
    # Construct file paths using os.path.join
    users_path = os.path.join(input_dir, 'u.user')
    movies_path = os.path.join(input_dir, 'u.item')
    ratings_path = os.path.join(input_dir, 'u.data')
    genre_path = os.path.join(input_dir, 'u.genre')

    # Load datasets
    users = pd.read_csv(users_path, sep='|', header=None,
                        names=['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code'])
    movies = pd.read_csv(movies_path, sep='|', header=None, encoding='latin-1',
                         names=['MovieID', 'Title', 'Release Date', 'Video Release Date', 'IMDb URL'] +
                         [f'Genre{i}' for i in range(19)])
    ratings = pd.read_csv(ratings_path, sep='\t', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    genre_mapping = pd.read_csv(genre_path, sep='|', header=None,
                                names=['Genre', 'Index']).dropna().set_index('Index')['Genre'].to_dict()
    genre_columns = list(genre_mapping.values())

    # Map genres to consistent columns
    genre_matrix = movies.iloc[:, 5:]
    genre_matrix.columns = genre_columns
    movies = movies[['MovieID', 'Title']].join(genre_matrix.reindex(columns=common_genres, fill_value=0))

    # Merge datasets
    data = pd.merge(pd.merge(ratings, users, on='UserID'), movies, on='MovieID')

    # Assign age groups
    data['Age Group'] = data['Age'].apply(
        lambda age: '18-24' if age <= 24 else
                    '25-34' if age <= 34 else
                    '35-44' if age <= 44 else
                    '45-49' if age <= 49 else
                    '50-54' if age <= 54 else
                    '55+'
    )

    # Display age group counts
    print("Age Group Counts:\n", data['Age Group'].value_counts())

    # Calculate sum of genre ratings per age group
    age_genre_ratings = data.groupby('Age Group')[common_genres].sum()

    # Normalize genre ratings for each age group
    avg_distributions = age_genre_ratings.div(age_genre_ratings.sum(axis=1), axis=0).fillna(0)

    return {age_group: avg_distributions.loc[age_group].tolist() for age_group in avg_distributions.index}
