import sys
import click
import joblib
from matplotlib import pyplot as plt
import numpy as np
import scipy
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw


def calculate_hull(
        X, 
        scale=1.1, 
        padding="scale", 
        n_interpolate=100, 
        interpolation="quadratic_periodic", 
        return_hull_points=False):
    """
    Calculates a "smooth" hull around given points in `X`.
    The different settings have different drawbacks but the given defaults work reasonably well.
    Parameters
    ----------
    X : np.ndarray
        2d-array with 2 columns and `n` rows
    scale : float, optional
        padding strength, by default 1.1
    padding : str, optional
        padding mode, by default "scale"
    n_interpolate : int, optional
        number of interpolation points, by default 100
    interpolation : str or callable(ix,iy,x), optional
        interpolation mode, by default "quadratic_periodic"

    Inspired by: https://stackoverflow.com/a/17557853/991496
    """
    
    if padding == "scale":

        # scaling based padding
        scaler = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(with_std=False),
            sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1)))
        points_scaled = scaler.fit_transform(X) * scale
        hull_scaled = scipy.spatial.ConvexHull(points_scaled, incremental=True)
        hull_points_scaled = points_scaled[hull_scaled.vertices]
        hull_points = scaler.inverse_transform(hull_points_scaled)
        hull_points = np.concatenate([hull_points, hull_points[:1]])
    
    elif padding == "extend" or isinstance(padding, (float, int)):
        # extension based padding
        # TODO: remove?
        if padding == "extend":
            add = (scale - 1) * np.max([
                X[:,0].max() - X[:,0].min(), 
                X[:,1].max() - X[:,1].min()])
        else:
            add = padding
        points_added = np.concatenate([
            X + [0,add], 
            X - [0,add], 
            X + [add, 0], 
            X - [add, 0]])
        hull = scipy.spatial.ConvexHull(points_added)
        hull_points = points_added[hull.vertices]
        hull_points = np.concatenate([hull_points, hull_points[:1]])
    else:
        raise ValueError(f"Unknown padding mode: {padding}")
    
    # number of interpolated points
    nt = np.linspace(0, 1, n_interpolate)
    
    x, y = hull_points[:,0], hull_points[:,1]
    
    # ensures the same spacing of points between all hull points
    t = np.zeros(x.shape)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t /= t[-1]

    # interpolation types
    if interpolation is None or interpolation == "linear":
        x2 = scipy.interpolate.interp1d(t, x, kind="linear")(nt)
        y2 = scipy.interpolate.interp1d(t, y, kind="linear")(nt)
    elif interpolation == "quadratic":
        x2 = scipy.interpolate.interp1d(t, x, kind="quadratic")(nt)
        y2 = scipy.interpolate.interp1d(t, y, kind="quadratic")(nt)

    elif interpolation == "quadratic_periodic":
        x2 = scipy.interpolate.splev(nt, scipy.interpolate.splrep(t, x, per=True, k=4))
        y2 = scipy.interpolate.splev(nt, scipy.interpolate.splrep(t, y, per=True, k=4))
    
    elif interpolation == "cubic":
        x2 = scipy.interpolate.CubicSpline(t, x, bc_type="periodic")(nt)
        y2 = scipy.interpolate.CubicSpline(t, y, bc_type="periodic")(nt)
    else:
        x2 = interpolation(t, x, nt)
        y2 = interpolation(t, y, nt)
    
    X_hull = np.concatenate([x2.reshape(-1,1), y2.reshape(-1,1)], axis=1)
    if return_hull_points:
        return X_hull, hull_points
    else:
        return X_hull


def get_color_map(groups, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=0, vmax=len(groups) - 1)
    return {group: cmap(norm(i)) for i, group in enumerate(groups)}


def create_tsne_image_grid_with_hulls(
    tsne_results,
    image_paths,
    db,
    output_size=(9000, 9000),
    thumbnail_size=(256, 256),
    scale=1.25, 
    padding="scale", 
    n_interpolate=100, 
    interpolation="quadratic_periodic", 
    cmap_name='tab20'
):
    """
    Create a large image grid based on t-SNE results.
    
    :param tsne_results: numpy array of shape (n_samples, 2) containing t-SNE results
    :param image_paths: list of paths to the image files
    :param output_size: tuple of (width, height) for the output image
    :param thumbnail_size: tuple of (width, height) for each thumbnail image
    :return: PIL Image object of the resulting grid
    """
    # Normalize t-SNE results to [0, 1] range
    x_norm = (tsne_results[:, 0] - tsne_results[:, 0].min()) / (tsne_results[:, 0].max() - tsne_results[:, 0].min())
    y_norm = (tsne_results[:, 1] - tsne_results[:, 1].min()) / (tsne_results[:, 1].max() - tsne_results[:, 1].min())

    canvas = Image.new('RGBA', output_size, color=(255, 255, 255, 255))

    # draw images on top
    for i, img_path in enumerate(image_paths):
        try:
            # Open and resize the image
            img = Image.open(img_path).convert('RGBA')
            img.thumbnail(thumbnail_size, Image.LANCZOS)

            # Calculate position
            x_pos = int(x_norm[i] * (output_size[0] - thumbnail_size[0]))
            y_pos = int(y_norm[i] * (output_size[1] - thumbnail_size[1]))

            # Paste the thumbnail onto the canvas
            canvas.alpha_composite(img, (x_pos, y_pos))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    # draw each cluster hull
    xynorm = np.stack([x_norm, y_norm], axis=1)
    points_by_cluster = {cluster: xynorm[db.labels_ == cluster] for cluster in set(db.labels_)}
    color_map = get_color_map(set(db.labels_), cmap_name=cmap_name)

    # Create a separate transparent layer for the polygons
    polygon_layer = Image.new('RGBA', output_size, color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(polygon_layer)

    for cluster in points_by_cluster:
        if cluster != -1:
            hull = calculate_hull(points_by_cluster[cluster], scale=scale, padding=padding, n_interpolate=n_interpolate, interpolation=interpolation)
            hull = hull * np.array(output_size)
            hull = hull.astype(int)
            color = tuple(int(x * 255) for x in color_map[cluster])
            draw.polygon([tuple(p) for p in hull], outline=color, fill=(*color[:3], 32), width=30)

    # Composite the polygon layer onto the main canvas
    canvas = Image.alpha_composite(canvas, polygon_layer)


    return canvas



@click.command()
@click.option('--input', type=click.Path(), required=True)
@click.option('--output', type=click.Path(), required=True)
def cli(input: str, output: str):
    try:
        df = joblib.load(input)
    except Exception as e:
        print('Failed to load enron joblib')
        print(e)
        return sys.exit(1)

    print('t-SNE...')
    X = np.array(df['embedding'].tolist())
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(X)

    print('Clustering...')
    db = DBSCAN(eps=4.75, min_samples=30, metric='euclidean').fit(X_embedded)

    print('Creating image...')
    img = create_tsne_image_grid_with_hulls(X_embedded, df['final_path'], db)
    img.save(output)

    print(f'Saved image to {output}')


if __name__ == '__main__':
    cli()