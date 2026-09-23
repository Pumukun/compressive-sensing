import numpy as np
import cv2


# Element budget for a batch of Gram matrices, the (W, P, P) stack.
# 8e6 float64 elements is roughly 64 MB.
MAX_GRAM_ELEMENTS = 8_000_000


def read_image(image_path: str, color: bool = False) -> np.ndarray:
    '''
    Read an image: (H, W) grayscale, or (H, W, 3) BGR when color=True.
    '''
    image = cv2.imread(image_path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f'could not read image: {image_path}')

    return image


def to_column_block(image: np.ndarray):
    '''
    Reshape an image into a column block (H, W * C) plus the channel count C.

    Channels are appended side by side so a colour image is solved by the same
    batched pass as a grayscale one, sharing one measurement and Gram matrix.
    '''
    if image.ndim == 2:
        return image.astype(np.float64), 1

    height, width, channels = image.shape
    block = image.transpose(0, 2, 1).reshape(height, width * channels)

    return block.astype(np.float64), channels


def from_column_block(block: np.ndarray, channels: int) -> np.ndarray:
    '''Inverse of to_column_block.'''
    if channels == 1:
        return block

    height, total = block.shape
    return block.reshape(height, channels, total // channels).transpose(0, 2, 1)


def lstsq(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Solve min ||A x - y||. Same minimum-norm solution as pinv(A) @ y, without
    forming the pseudo-inverse.
    '''
    solution, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return solution


def gram_chunk_size(total_columns: int, support_size: int) -> int:
    '''How many columns to process at once so the Gram stack fits in memory.'''
    if support_size <= 0:
        return max(1, total_columns)
    return max(1, min(total_columns, MAX_GRAM_ELEMENTS // (support_size * support_size)))


def solve_stack(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    '''
    Batched solve of gram[i] @ x[i] = rhs[i] over a (W, k, k) stack.

    Falls back to solving one by one if any sub-matrix is singular, so a single
    ill-conditioned column cannot fail the whole image.
    '''
    try:
        return np.linalg.solve(gram, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        pass

    solution = np.empty_like(rhs)
    for i in range(gram.shape[0]):
        try:
            solution[i] = np.linalg.solve(gram[i], rhs[i])
        except np.linalg.LinAlgError:
            solution[i] = lstsq(gram[i], rhs[i][:, None])[:, 0]

    return solution


def top_k_per_column(values: np.ndarray, k: int) -> np.ndarray:
    '''
    Indices of the k largest elements in every column, shape (W, k), unordered.

    Partitions rows of a transposed copy: argpartition along the non-contiguous
    axis of a C-ordered array costs about twice as much.
    '''
    transposed = np.ascontiguousarray(values.T)
    columns, rows = transposed.shape

    if k >= rows:
        return np.tile(np.arange(rows, dtype=np.intp), (columns, 1))

    return np.argpartition(transposed, -k, axis=1)[:, -k:]


def top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    '''
    Indices of the k largest elements by value, unordered.

    argpartition rather than argsort: only the set is needed, so O(N) instead
    of O(N log N).
    '''
    flat = np.asarray(values).ravel()

    if k <= 0:
        return np.empty(0, dtype=np.intp)
    if k >= flat.size:
        return np.arange(flat.size, dtype=np.intp)

    return np.argpartition(flat, -k)[-k:]


def to_uint8(image: np.ndarray) -> np.ndarray:
    '''
    Convert a reconstructed image to uint8: clip to [0, 255] and round.

    Clipping matters: a bare astype(np.uint8) wraps modulo 256.
    '''
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


class ImageCS():
    '''
    Container for an image together with its compression metrics.

    Attributes:
        __matrix (np.ndarray): image data
        __cr (float): compression ratio
        __psnr (float): peak signal-to-noise ratio
        __ssim (float): structural similarity index
    '''
    def __init__(self, matrix=np.ndarray((0,0)), cr: float=0.0, psnr: float=0.0, ssim: float=0.0):
        self.__matrix = matrix
        self.__cr: float = cr
        self.__psnr: float = psnr
        self.__ssim: float = ssim

    def get_Image(self) -> np.ndarray:
        return self.__matrix

    def get_CR(self) -> float:
        return self.__cr

    def get_PSNR(self) -> float:
        return self.__psnr

    def get_SSIM(self) -> float:
        return self.__ssim


    def set_Image(self, image: np.ndarray) -> None:
        self.__matrix = image

    def set_CR(self, cr) -> None:
        self.__cr = cr

    def set_PSNR(self, psnr) -> None:
        self.__psnr = psnr

    def set_SSIM(self, ssim) -> None:
        self.__ssim = ssim
