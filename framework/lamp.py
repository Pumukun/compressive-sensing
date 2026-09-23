import numpy as np
import framework.metrics as metrics

from framework.measurement import resolve as resolve_measurement
from framework.utils import (ImageCS, from_column_block, lstsq, read_image,
                             to_column_block, to_uint8, top_k_per_column)
from typing import Optional

try:
    import maxflow
except ImportError:  # pragma: no cover
    maxflow = None


def lamp(image_path: str, matrix: np.ndarray, M: int, K: int, seed: Optional[int] = None,
         color: bool = False, measurement='gaussian', block_width: int = 2) -> ImageCS:
    '''
    Lattice Matching Pursuit over a 2D image.
        image_path - path to the image (grayscale or colour).
        matrix - NxN basis matrix.
        M - number of measurements.
        K - sparsity level per column.
        seed - RNG seed for the measurement matrix (None means random).
        color - process the image as colour.
        measurement - 'gaussian', 'bernoulli', 'hadamard', or a ready-made MxN matrix.
        block_width - columns per lattice block; 1 disables the spatial term.

    Requires the maxflow package.
    '''
    image = read_image(image_path, color)

    im, channels = to_column_block(image)
    N = im.shape[0]

    Phi = resolve_measurement(measurement, M, N, seed)
    img_cs_1d = np.dot(Phi, im)

    Theta_1d = np.dot(Phi, matrix)

    sparse_rec_1d = cs_lamp_columns(img_cs_1d, Theta_1d, K, block_width=block_width)

    img_rec = to_uint8(from_column_block(np.dot(matrix, sparse_rec_1d), channels))

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)
    SSIM: float = metrics.SSIM(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR, ssim=SSIM)

    return img_res


def cs_lamp_columns(Y: np.ndarray, Phi: np.ndarray, K: int, block_width: int = 2,
                    max_iter: int = 15, tol: float = 1e-3, sigma: float = 1.0,
                    pairwise_weight: float = 1.0) -> np.ndarray:
    '''
    LaMP over every column of the image, in lattice blocks.
        Y - MxW measurement matrix, one column per image column.
        Phi - MxN product of the basis and the measurement matrix.
        K - sparsity level per column.
        block_width - columns per lattice block.

    The other algorithms pick a support by coefficient magnitude alone. LaMP
    assumes significant coefficients cluster spatially, and picks the support by
    minimising an energy over a lattice of adjacent columns: a unary term from
    the coefficient magnitude, and a pairwise term that penalises isolated
    selections. The minimum is found exactly by graph cut.

    The measurement model is the one the rest of the framework uses, so a block
    of adjacent columns shares Phi and the least-squares step stays per column.
    Only the support selection is joint.
    '''
    if maxflow is None:
        raise ImportError('LaMP requires the maxflow package: pip install PyMaxflow')

    M, N = Phi.shape
    W = Y.shape[1]

    K = max(0, min(K, M, N))

    X = np.zeros((N, W))
    if K == 0 or W == 0:
        return X

    width = max(1, block_width)

    for start in range(0, W, width):
        stop = min(start + width, W)
        X[:, start:stop] = _lamp_block(Y[:, start:stop], Phi, K, max_iter, tol,
                                       sigma, pairwise_weight)

    return X


def _lamp_block(Y: np.ndarray, Phi: np.ndarray, K: int, max_iter: int, tol: float,
                sigma: float, pairwise_weight: float) -> np.ndarray:
    '''One block of adjacent columns.'''
    M, N = Phi.shape
    w = Y.shape[1]

    S = np.zeros((N, w))
    target = np.linalg.norm(Y) * tol

    for _ in range(max_iter):
        residual = Y - np.dot(Phi, S)
        proposal = S + np.dot(Phi.T, residual)

        selected = _select_support(proposal, K, M, sigma, pairwise_weight)

        S_new = np.zeros((N, w))
        for j in range(w):
            index = np.flatnonzero(selected[:, j])
            if index.size:
                S_new[index, j] = lstsq(Phi[:, index], Y[:, j:j + 1]).ravel()

        S_new = _prune(S_new, K)

        if np.linalg.norm(Y - np.dot(Phi, S_new)) < target:
            return S_new
        S = S_new

    return S


def _select_support(proposal: np.ndarray, K: int, M: int, sigma: float,
                    pairwise_weight: float) -> np.ndarray:
    '''
    Minimise the lattice energy by graph cut; returns a boolean (N, w) mask.

    Both unary costs are expressed relative to the threshold tau, so the energy
    does not depend on the overall scale of the coefficients. Keeping is cheap
    once the magnitude passes tau, discarding grows costly beyond it. The
    pairwise term penalises neighbours taking different labels, which is
    submodular and therefore exactly minimisable.
    '''
    N, w = proposal.shape
    magnitude = np.abs(proposal)
    tau = _threshold(magnitude, K * w)

    if tau <= 0:
        return magnitude > 0

    ratio = magnitude / tau
    keep_cost = 1.0 - ratio                            # cost of label 1
    drop_cost = (ratio / max(sigma, 1e-12)) ** 2 / 2.0 # cost of label 0

    # Shift per node, not globally: a global shift would change the balance
    # between the two labels, since only their difference matters
    base = np.minimum(keep_cost, drop_cost)
    keep_cost -= base
    drop_cost -= base

    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes((N, w))
    graph.add_grid_edges(nodes, pairwise_weight)
    # add_grid_tedges(nodes, source_capacity, sink_capacity): cutting the source
    # edge costs source_capacity and puts the node in the sink segment, so the
    # source capacity carries the cost of label 1.
    graph.add_grid_tedges(nodes, keep_cost, drop_cost)
    graph.maxflow()

    selected = graph.get_grid_segments(nodes)

    # The least-squares step must stay overdetermined: a support approaching M
    # interpolates the measurements instead of fitting them
    return _cap_support(selected, magnitude, max(K, min(M // 2, 5 * K)))


def _cap_support(selected: np.ndarray, magnitude: np.ndarray, limit: int) -> np.ndarray:
    '''Trim any column selecting more than limit atoms to its largest ones.'''
    capped = selected.copy()

    for j in range(selected.shape[1]):
        index = np.flatnonzero(selected[:, j])
        if index.size <= limit:
            continue
        order = np.argpartition(magnitude[index, j], -limit)[-limit:]
        capped[:, j] = False
        capped[index[order], j] = True

    return capped


def _threshold(magnitude: np.ndarray, count: int) -> float:
    '''Magnitude of the 5*count-th largest coefficient, the selection threshold.'''
    flat = magnitude.ravel()
    take = int(min(max(5 * count, 1), flat.size))
    return float(np.partition(flat, -take)[-take])


def _prune(S: np.ndarray, K: int) -> np.ndarray:
    '''Keep the K largest coefficients in every column.'''
    N, w = S.shape
    if K >= N:
        return S

    keep = top_k_per_column(np.abs(S), K)
    pruned = np.zeros_like(S)
    cols = np.arange(w)
    pruned[keep.T, cols] = S[keep.T, cols]

    return pruned
