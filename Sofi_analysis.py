import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import map_coordinates
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

folder = Path(r"C:Frames_for_calculation")
files = sorted(folder.glob("*.png"))

# SOFI processing parameters.
N_SERIES = 40  # Number of excitation series per source block.
SOFI_SERIES_MODE = 40  # Number of excitation series per analysis block: 20 or 40.

if SOFI_SERIES_MODE not in (20, 40):
    raise ValueError("SOFI_SERIES_MODE should be 20 or 40")

if N_SERIES % SOFI_SERIES_MODE != 0:
    raise ValueError("N_SERIES must be divisible by SOFI_SERIES_MODE without remainder")

N_DELAY = 20
FAST_END = 5
TOTAL_END = 20  # Fast channel: frames 0–4; slow channel: frames 5–19.
COUNT_THRESHOLD = 30.0
SIGNAL_THRESHOLD = 20.0

# Allowed interval between consecutive series starts.
SERIES_GAP_MIN = 20
SERIES_GAP_MAX = 30

# The first frame after each signal is empty.
EMPTY_FRAME_OFFSET = 1
DATA_FRAME_OFFSET = 2  # Offset of the first working delay frame.

# Thread counts for frame loading and signal detection.
MAX_WORKERS = 4
SIGNAL_WORKERS = 8

# Cache settings for repeated processing and plotting.
REBUILD_STARTS_CACHE = True
REBUILD_FRAME_CACHE = True
CACHE_DIR = folder / "_sofi_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Update the cache version when processing settings or logic change.
CACHE_TAG = (
    f"source_s{N_SERIES}"
    f"_analysis_s{SOFI_SERIES_MODE}"
    f"_d{N_DELAY}"
    f"_f{FAST_END}_t{TOTAL_END}"
    f"_h424_w330"
    f"_empty{EMPTY_FRAME_OFFSET}"
    f"_v7"
)

STARTS_CACHE = CACHE_DIR / (
    f"starts_s{N_SERIES}"
    f"_sig{SIGNAL_THRESHOLD:g}"
    f"_gap{SERIES_GAP_MIN}-{SERIES_GAP_MAX}"
    f"_localmean15_v4.npy"
)
F1_CACHE = CACHE_DIR / f"F1_{CACHE_TAG}.npy"
F2_CACHE = CACHE_DIR / f"F2_{CACHE_TAG}.npy"
DIAG_CACHE = CACHE_DIR / f"diagnostics_{CACHE_TAG}.npz"

if not files:
    raise FileNotFoundError(f"PNG-files not found: {folder}")

file_by_number = {int(path.stem): path for path in files}
# Emitter coordinates in the cropped image.

points = [  # Emitter coordinates (x, y, label).
    (163, 95, "1"),
    (86, 180, "2"),
    (127, 235, "3"),
    (70, 255, "4"),
    (253, 245, "5"),
    (205, 315, "6"),
    (133, 168, "7"),
    (198, 180, "8"),
    (175, 250, "9"),
    (100, 320, "10"),
    (153, 330, "11"),
    (260, 353, "12"),
]

# Parameter calibration and map reconstruction.


def zeta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    tau = 6.5 - 3.5 * theta

    Y_fast = 1.0 - np.exp(-3.8 / tau)
    Y_total = 1.0 - np.exp(-20 / tau)

    return Y_fast / Y_total


# Lookup table for numerical inversion of zeta.
theta_grid = np.linspace(-1.5, 1.5, 20001)
zeta_grid = zeta(theta_grid)

zeta_min = zeta_grid[0]
zeta_max = zeta_grid[-1]


def inverse_zeta(Z):
    Z = np.asarray(Z, dtype=np.float64)

    theta = np.full_like(Z, np.nan)

    valid = np.isfinite(Z) & (Z >= zeta_min) & (Z <= zeta_max)

    theta[valid] = np.interp(Z[valid], zeta_grid, theta_grid)
    return theta


@lru_cache(maxsize=8)
def _regularization_matrix(H, W, lam):
    # Cache the regularization matrix for reuse across cumulant orders.
    Dx = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(W, W), format="csr")
    Dy = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(H, H), format="csr")
    Ix = sparse.eye(W, format="csr")
    Iy = sparse.eye(H, format="csr")
    L = sparse.kron(Iy, Dx, format="csr") + sparse.kron(Dy, Ix, format="csr")
    return float(lam) * (L.T @ L)


def smooth_theta_map(theta_obs, weight, lam=50.0):
    H, W = theta_obs.shape
    weight_flat = weight.ravel().astype(np.float64, copy=False)

    A = sparse.diags(weight_flat, format="csr")
    A = A + _regularization_matrix(H, W, float(lam))
    b = weight_flat * theta_obs.ravel()

    theta_flat = spsolve(A, b)
    return theta_flat.reshape(H, W)


def decode_png(path):
    dat = np.fromfile(path, dtype=np.uint8)
    frame = cv2.imdecode(dat, cv2.IMREAD_UNCHANGED)

    if frame is None:
        raise OSError(f"Cannot decode {path}")

    return frame


def signal_score(path):

    # Return the maximum local mean brightness in the signal-marker region.
    # Average over 15 × 15 pixels to suppress isolated bright pixels.

    frame = decode_png(path)

    roi = frame[210:330, 0:100].astype(np.float32)

    local_mean = cv2.blur(roi, ksize=(15, 15), borderType=cv2.BORDER_REPLICATE)

    return float(np.max(local_mean))


def extract_signal_frames(signal_strength, frame_numbers, threshold):

    # Merge consecutive above-threshold frames into one physical flash.
    # Select the frame with the highest signal score in each group.

    signal_strength = np.asarray(signal_strength, dtype=np.float32)

    frame_numbers = np.asarray(frame_numbers, dtype=np.int64)

    above = np.flatnonzero(signal_strength >= threshold)

    if above.size == 0:
        raise RuntimeError("Not a single signal frame is found")

    above_frame_numbers = frame_numbers[above]

    # A new group starts if there is at least one frame below the threshold between the selected frames.
    split_positions = np.where(np.diff(above_frame_numbers) > 1)[0] + 1

    groups = np.split(above, split_positions)

    peak_indices = []

    for group in groups:
        local_index = np.argmax(signal_strength[group])

        peak_indices.append(group[local_index])

    peak_indices = np.asarray(peak_indices, dtype=np.int64)

    signal_frames = frame_numbers[peak_indices]

    return signal_frames, peak_indices


def read_crop(frame_number):
    # Read the working region of one frame.
    path = file_by_number.get(int(frame_number))

    if path is None:
        raise FileNotFoundError(f"Frame {int(frame_number):06d}.png not found")

    frame = decode_png(path)
    return frame[:, 240:570]


def calculate_starts():

    # Detect signal flashes and construct complete source blocks
    # containing N_SERIES consecutive excitation series.

    if STARTS_CACHE.exists() and not REBUILD_STARTS_CACHE:
        starts = np.load(STARTS_CACHE)
        return starts

    frame_numbers = np.asarray([int(path.stem) for path in files], dtype=np.int64)

    with ThreadPoolExecutor(max_workers=SIGNAL_WORKERS) as executor:

        signal_strength = np.fromiter(
            executor.map(signal_score, files), dtype=np.float32, count=len(files)
        )

    background = float(np.median(signal_strength))

    absolute_deviation = np.abs(signal_strength - background)

    mad = float(1.4826 * np.median(absolute_deviation))

    adaptive_threshold = background + 6.0 * mad

    used_threshold = max(float(SIGNAL_THRESHOLD), adaptive_threshold)

    print("used signal threshold:", used_threshold)

    sig_n, signal_peak_indices = extract_signal_frames(
        signal_strength=signal_strength,
        frame_numbers=frame_numbers,
        threshold=used_threshold,
    )

    if len(sig_n) < N_SERIES + 1:
        raise RuntimeError("Too small number of signal frames")

    signal_gaps = np.diff(sig_n)

    # Indices of the second flash in each candidate double start marker.
    candidate_indices = np.where((signal_gaps >= 1) & (signal_gaps <= 3))[0] + 1

    sig_scores = signal_strength[signal_peak_indices]

    complete_blocks = []
    last_accepted_end = -1

    for candidate_index in candidate_indices:

        start_frame = sig_n[candidate_index]

        if start_frame <= last_accepted_end:
            continue

        block = [start_frame]
        current_index = candidate_index
        block_is_valid = True

        for _ in range(N_SERIES - 1):

            minimum_frame = block[-1] + SERIES_GAP_MIN
            maximum_frame = block[-1] + SERIES_GAP_MAX

            possible_indices = np.where(
                (sig_n[current_index + 1 :] >= minimum_frame)
                & (sig_n[current_index + 1 :] <= maximum_frame)
            )[0]

            possible_indices += current_index + 1

            if possible_indices.size == 0:
                block_is_valid = False
                break

            next_index = possible_indices[np.argmax(sig_scores[possible_indices])]

            block.append(sig_n[next_index])
            current_index = next_index

        if not block_is_valid:
            continue

        block = np.asarray(block, dtype=np.int64)
        gaps = np.diff(block)

        if np.all((gaps >= SERIES_GAP_MIN) & (gaps <= SERIES_GAP_MAX)):
            complete_blocks.append(block)
            last_accepted_end = block[-1]

    if not complete_blocks:
        raise RuntimeError(
            "Not a single SOFI block is found"
            "Check SIGNAL_THRESHOLD value and intervals between signal frames"
        )

    starts = np.stack(complete_blocks, axis=0).astype(np.int64)

    # Validate the intervals within the accepted source blocks.
    invalid_blocks = np.any(
        (np.diff(starts, axis=1) < SERIES_GAP_MIN)
        | (np.diff(starts, axis=1) > SERIES_GAP_MAX),
        axis=1,
    )

    if np.any(invalid_blocks):
        raise RuntimeError(
            "After forming array of the starts, there are leftover blocks with wrong intervals"
        )

    np.save(STARTS_CACHE, starts)
    return starts


def prepare_analysis_starts(starts_40, series_per_sofi):

    # Prepare series start indices for analysis.
    # Use either 40 or 20 excitation series per analysis block.
    # In mode 20, split each source block into two consecutive groups.

    starts_40 = np.asarray(starts_40, dtype=np.int64)

    if starts_40.ndim != 2:
        raise ValueError("starts should be two-dimentional array")

    if starts_40.shape[1] != N_SERIES:
        raise ValueError(
            f"Expected {N_SERIES} series in the initial block, "
            f"received {starts_40.shape[1]}"
        )

    if series_per_sofi == N_SERIES:
        return starts_40.copy()

    if N_SERIES % series_per_sofi != 0:
        raise ValueError(
            "Initial block cannot be devided " "on parts of requested size"
        )

    number_of_parts = N_SERIES // series_per_sofi

    analysis_starts = starts_40.reshape(
        starts_40.shape[0] * number_of_parts, series_per_sofi
    )

    return np.ascontiguousarray(analysis_starts, dtype=np.int64)


def load_sofi_block(starts_row, executor):
    # Load empty frames and delay frames for one analysis block
    starts_row = np.asarray(starts_row, dtype=np.int64)

    if starts_row.ndim != 1:
        raise ValueError("starts_row must be one-dimensional")

    n_series = starts_row.size
    if n_series not in (20, 40):
        raise ValueError(f"Expected 20 or 40 series, got {n_series}")

    empty_numbers = starts_row + EMPTY_FRAME_OFFSET
    delay_offsets = DATA_FRAME_OFFSET + np.arange(N_DELAY, dtype=np.int64)
    data_numbers = starts_row[:, None] + delay_offsets[None, :]
    all_numbers = np.concatenate((empty_numbers, data_numbers.ravel()))

    if executor is None:
        loaded = [read_crop(number) for number in all_numbers]
    else:
        loaded = list(executor.map(read_crop, all_numbers))

    loaded = np.stack(loaded)
    height, width = loaded.shape[1:]

    empty_frames = loaded[:n_series]
    delay_frames = loaded[n_series:].reshape(n_series, N_DELAY, height, width)
    return empty_frames, delay_frames


def calculate_cumulants_chunked(X1, X2=None, chunk_size=16):
    # Calculate cumulants 1–6 in internal order [1, 2, 4, 6, 3, 5].
    N, H, W = X1.shape

    mean = np.zeros((H, W), dtype=np.float64)

    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        block = X1[start:stop].astype(np.float64)

        if X2 is not None:
            block += X2[start:stop]

        mean += np.sum(block, axis=0)

    mean /= N

    sums = [np.zeros((H, W), dtype=np.float64) for _ in range(5)]

    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        block = X1[start:stop].astype(np.float64)

        if X2 is not None:
            block += X2[start:stop]

        d = block - mean[None, :, :]
        p2 = d * d
        p3 = p2 * d
        p4 = p3 * d
        p5 = p4 * d
        p6 = p5 * d

        sums[0] += np.sum(p2, axis=0)
        sums[1] += np.sum(p3, axis=0)
        sums[2] += np.sum(p4, axis=0)
        sums[3] += np.sum(p5, axis=0)
        sums[4] += np.sum(p6, axis=0)

    mu2, mu3, mu4, mu5, mu6 = [value / N for value in sums]

    kappa4 = mu4 - 3.0 * mu2**2
    kappa5 = mu5 - 10.0 * mu3 * mu2
    kappa6 = mu6 - 15.0 * mu4 * mu2 - 10.0 * mu3**2 + 30.0 * mu2**3

    return np.stack((mean, mu2, kappa4, kappa6, mu3, kappa5), axis=0)


print("start of calculation", datetime.now())
starts_40 = calculate_starts()

starts = prepare_analysis_starts(starts_40, SOFI_SERIES_MODE)

N = starts.shape[0]

H = 424  # Height of the cropped frame.
W = 330

if (
    F1_CACHE.exists()
    and F2_CACHE.exists()
    and DIAG_CACHE.exists()
    and not REBUILD_FRAME_CACHE
):

    F1 = np.load(F1_CACHE, mmap_mode="r")
    F2 = np.load(F2_CACHE, mmap_mode="r")

    diagnostics = np.load(DIAG_CACHE)
    delay_counts = diagnostics["delay_counts"]
    delay_intensity = diagnostics["delay_intensity"]
    weighted_fast_energy = diagnostics["weighted_fast_energy"]
    weighted_total_energy = diagnostics["weighted_total_energy"]

else:
    F1 = np.empty((N, H, W), dtype=np.float32)
    F2 = np.empty((N, H, W), dtype=np.float32)

    delay_counts = np.zeros((len(points), N_DELAY), dtype=np.int64)
    delay_intensity = np.zeros((len(points), N_DELAY), dtype=np.float64)

    weighted_fast_energy = np.zeros(len(points), dtype=np.float64)
    weighted_total_energy = np.zeros(len(points), dtype=np.float64)

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS) if MAX_WORKERS > 1 else None

    try:
        for i in range(N):
            empty_frames, delay_frames = load_sofi_block(starts[i], executor)

            # Exclude empty frames from both channels.
            # Estimate the baseline from the median of the empty frames in this block.
            baseline = np.median(empty_frames.astype(np.float32), axis=0)

            signal = delay_frames.astype(np.float32) - baseline[None, None, :, :]

            fast = np.sum(signal[:, :FAST_END], axis=(0, 1), dtype=np.float64)

            slow = np.sum(signal[:, FAST_END:TOTAL_END], axis=(0, 1), dtype=np.float64)

            F1[i] = fast
            F2[i] = slow

            # Collect diagnostics from the central emitter regions.
            for emitter, (x, y, number) in enumerate(points):
                roi_count = signal[:, :, y - 1 : y + 2, x - 1 : x + 2]

                trace_count = np.mean(roi_count, axis=(2, 3))

                peak_delay = np.argmax(trace_count, axis=1)
                peak_value = np.max(trace_count, axis=1)
                valid = peak_value > COUNT_THRESHOLD

                delay_counts[emitter] += np.bincount(
                    peak_delay[valid], minlength=N_DELAY
                )

                radius = 2
                roi_intensity = signal[
                    :, :, y - radius : y + radius + 1, x - radius : x + radius + 1
                ]

                trace_intensity = np.sum(roi_intensity, axis=(2, 3), dtype=np.float64)

                delay_intensity[emitter] += np.sum(trace_intensity, axis=0)

                event_energy = np.sum(trace_intensity[:, :TOTAL_END], axis=1)

                valid_events = valid & (peak_delay < TOTAL_END)

                weighted_fast_energy[emitter] += np.sum(
                    event_energy[valid_events & (peak_delay < FAST_END)]
                )

                weighted_total_energy[emitter] += np.sum(event_energy[valid_events])

    finally:
        if executor is not None:
            executor.shutdown()

    np.save(F1_CACHE, F1)
    np.save(F2_CACHE, F2)
    np.savez(
        DIAG_CACHE,
        delay_counts=delay_counts,
        delay_intensity=delay_intensity,
        weighted_fast_energy=weighted_fast_energy,
        weighted_total_energy=weighted_total_energy)

    print("preprocessed arrays saved to:", CACHE_DIR)


for emitter in range(len(points)):
    profile = delay_intensity[emitter, :TOTAL_END].copy()
    total = np.sum(profile)

    if total != 0:
        profile /= total

    plt.plot(np.arange(TOTAL_END), profile, marker="o", label=str(emitter + 1))

plt.xlabel("Delay frame")
plt.ylabel("Normalized integrated intensity")
plt.legend()
plt.grid()
plt.show()

# Calculate channel and total-signal cumulants.
C = np.zeros((6, 2, H, W), dtype=np.float64)
C[:, 0] = calculate_cumulants_chunked(F1)
C[:, 1] = calculate_cumulants_chunked(F2)
FullC = calculate_cumulants_chunked(F1, F2)

print("cumulants finished", datetime.now())

# Reconstruct the parameter maps.

print("theta calculating")

orders = [1, 2, 4, 6, 3, 5]

H, W = C.shape[2:]

theta_local = np.full((6, H, W), np.nan)
theta_map = np.full((6, H, W), np.nan)
eps = 1e-12

for i, n in enumerate(orders):

    C_fast = C[i, 0].astype(np.float64)
    C_total_n = FullC[i].astype(np.float64)

    # Calculate the cumulant ratio before taking the nth root.
    ratio = np.full((H, W), np.nan)

    total_scale = np.percentile(np.abs(C_total_n), 99)
    threshold = 0.01 * total_scale

    valid = (
        np.isfinite(C_fast) & np.isfinite(C_total_n) & (np.abs(C_total_n) > threshold)
    )

    ratio[valid] = C_fast[valid] / C_total_n[valid]

    # Both cumulants have the same sign in the ideal model.
    valid &= ratio > 0

    Z = np.full((H, W), np.nan)

    # Take the nth root to recover the channel fraction.
    Z[valid] = ratio[valid] ** (1.0 / n)

    # Invert the calibration to obtain the local parameter estimate.
    theta_local[i] = inverse_zeta(Z)

    valid_theta = np.isfinite(theta_local[i])

    # Assign confidence weights.
    weight = np.abs(C_total_n)
    weight[~valid_theta] = 0.0

    positive_weight = weight[weight > 0]

    if positive_weight.size == 0:
        print(f"n={n}: valid pixels not found")
        continue

    weight_scale = np.percentile(positive_weight, 99)

    weight /= weight_scale + eps
    weight = np.clip(weight, 0, 1)

    theta_for_smoothing = np.nan_to_num(theta_local[i], nan=0.0)

    theta_map[i] = smooth_theta_map(theta_for_smoothing, weight, lam=100)

trajectory_points = np.array(
    [(163, 0), (163, 95), (133, 168), (127, 235), (100, 320), (100, 420)], dtype=float
)

def make_trajectory(points, step=1.0):
    x_path = []
    y_path = []
    distance = []

    accumulated_distance = 0.0

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]

        segment_length = np.hypot(x1 - x0, y1 - y0)
        number = max(2, int(np.ceil(segment_length / step)) + 1)

        t = np.linspace(0, 1, number)

        if i > 0:
            t = t[1:]

        x_path.extend(x0 + t * (x1 - x0))
        y_path.extend(y0 + t * (y1 - y0))
        distance.extend(accumulated_distance + t * segment_length)

        accumulated_distance += segment_length

    return (np.asarray(distance), np.asarray(x_path), np.asarray(y_path))


x, x_path, y_path = make_trajectory(trajectory_points, step=1.0)

profiles = [
    map_coordinates(theta_map[i], [y_path, x_path], order=1, mode="nearest")
    for i in range(4)
]

images = [
    theta_map[0, :, :],  # n=1
    theta_map[1, :, :],  # n=2
    theta_map[4, :, :],  # n=3
    theta_map[2, :, :],  # n=4
    theta_map[5, :, :],  # n=5
    theta_map[3, :, :],  # n=6
]

orders = [1, 2, 3, 4, 5, 6]

# Cumulant orders for which the trajectory is displayed.
trajectory_orders = {1, 2, 4, 6}

letters = ["a", "b", "c", "d", "e", "f"]

fig, ax = plt.subplots(3, 2, figsize=(9, 6), facecolor="white")

plt.subplots_adjust(
    left=0.03, right=0.88, bottom=0.03, top=0.97, wspace=0.03, hspace=0.03
)

for a in ax.ravel():
    for x0, y0, number in points:
        a.scatter(
            x0,
            y0,
            s=30,
            marker="x",
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )

for i, (a, order) in enumerate(zip(ax.ravel(), orders)):
    im = a.imshow(images[i], cmap="seismic", vmin=-1.5, vmax=1.5, aspect="equal")

    a.set_xticks([])
    a.set_yticks([])
    a.set_box_aspect(H / W)

    if order in trajectory_orders:
        a.plot(x_path, y_path, linestyle=":", color="yellow", linewidth=1.5, zorder=12)

    for spine in a.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)

    a.text(
        0.03,
        0.97,
        letters[i],
        transform=a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )

cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, fraction=0.035, pad=0.02)

cbar.set_label(r"Value of estimated parameter $\theta$", fontsize=14)
cbar.set_ticks(np.linspace(-1.5, 1.5, 7))
plt.show()

H, W = C.shape[2:]
fig, ax = plt.subplots(3, 2, figsize=(5.3, 7.5))

plt.subplots_adjust(
    left=0.03, right=0.97, bottom=0.03, top=0.97, wspace=0.03, hspace=0.03
)

n = 0
rgb = np.zeros((H, W, 3))
plot_indices = [0, 1, 4, 2, 5, 3]

for index, a in zip(plot_indices, ax.ravel()):
    hot = np.abs(C[index, 0])  # Magnitude of the fast-channel cumulant.
    cold = np.abs(C[index, 1])  # Magnitude of the slow-channel cumulant.

    values = np.concatenate((hot.ravel(), cold.ravel()))
    vmax = np.percentile(values, 99.7)

    rgb[:, :, 0] = np.clip(hot / vmax, 0, 1)
    rgb[:, :, 2] = np.clip(cold / vmax, 0, 1)

    a.imshow(rgb, aspect="equal")
    a.set_box_aspect(H / W)

    a.set_xticks([])
    a.set_yticks([])

    for spine in a.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)

    a.text(
        0.03,
        0.97,
        letters[n],
        transform=a.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
    )
    n += 1

for a in ax.ravel():
    for x0, y0, number in points:
        a.text(
            x0,
            y0,
            number,
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
        )

plt.show()

x, x_path, y_path = make_trajectory(trajectory_points, step=1.0)

lw = 1  # Spine width in points.
colors = ["orange", "blue", "green", "magenta"]
linestyles = ["-.", "--", "-", "-"]

fig, ax = plt.subplots(figsize=(6, 4))

ax.set_yticks(np.linspace(-1, 1, 5))

ax.plot(
    x, profiles[0], color=colors[0], linestyle=linestyles[0], linewidth=2, label="n=1"
)
ax.plot(
    x, profiles[1], color=colors[1], linestyle=linestyles[1], linewidth=2, label="n=2"
)
ax.plot(
    x, profiles[2], color=colors[2], linestyle=linestyles[2], linewidth=2, label="n=4"
)
ax.plot(
    x, profiles[3], color=colors[3], linestyle=linestyles[3], linewidth=1, label="n=6"
)

# Style the plot frame.
for spine in ax.spines.values():
    spine.set_linewidth(lw)
    spine.set_color("black")

# Point the ticks inward.
ax.tick_params(
    axis="both",
    which="major",
    labelsize=11,
    direction="in",
    width=1,
    length=6,
    top=True,
    right=True,
)

ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.4)

ax.set_axisbelow(True)
ax.set_xlabel("Distance along trajectory of the crossection, pixels", fontsize=14)
ax.set_ylabel(r"Dimensionless temperature $\theta$", fontsize=14)
leg = ax.legend(fontsize=14, frameon=False, handlelength=3)

ax.scatter(
    73,
    1,
    s=25,
    marker="o",
    facecolor="black",
    edgecolor="black",
    linewidth=1.5,
    zorder=10,
)

ax.scatter(
    160,
    -1,
    s=25,
    marker="o",
    facecolor="black",
    edgecolor="black",
    linewidth=1.5,
    zorder=10,
)

ax.scatter(
    241,
    1,
    s=25,
    marker="o",
    facecolor="black",
    edgecolor="black",
    linewidth=1.5,
    zorder=10,
)

ax.scatter(
    330,
    -1,
    s=25,
    marker="o",
    facecolor="black",
    edgecolor="black",
    linewidth=1.5,
    zorder=10,
)

plt.tight_layout()
plt.show()

theta_true = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1], dtype=np.float64)

T_emi = np.zeros((len(points), 6))

for i, (x, y, n) in enumerate(points):
    T_emi[i, :] = theta_map[:, y, x]

mse_internal_order = np.mean((T_emi - theta_true[:, None]) ** 2, axis=0)

plot_order = [0, 1, 4, 2, 5, 3]
MSE = mse_internal_order[plot_order]

print("MSE:", MSE)

x = [1, 2, 3, 4, 5, 6]

lw = 1

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_yticks(np.linspace(0, 0.5, 5))
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.plot(x, MSE, marker="o", markersize=7, color="royalblue", linestyle="-", linewidth=2)

for spine in ax.spines.values():
    spine.set_linewidth(lw)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    labelsize=11,
    direction="in",
    width=1,
    length=6,
    top=True,
    right=True,
)

ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.4)

ax.set_axisbelow(True)
ax.set_xlabel("Order of cumulant", fontsize=14)
ax.set_ylabel("MSE", fontsize=14)
plt.show()

# Convert internal cumulant order [1, 2, 4, 6, 3, 5] to [1, 2, 3, 4, 5, 6].
table_indices = [0, 1, 4, 2, 5, 3]

header = (
    f"{'Emitter':>7} {'x':>5} {'y':>5}"
    + "".join(f"{f'n={n}':>10}" for n in range(1, 7))
)
print(header)
print("-" * len(header))

for x_coord, y_coord, label in points:
    temperatures = theta_map[table_indices, y_coord, x_coord]
    row = f"{label:>7} {x_coord:5d} {y_coord:5d}"
    row += "".join(f"{value:10.3f}" for value in temperatures)
    print(row)
