use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::{ArrayIndex, Ellipsis, IndexOp, TryIndexOp};
use mlx_rs::Array;

/// Manual scatter-add for 1-D arrays: result[indices[i]] += updates[i].
///
/// Equivalent to `scatter_add_single(a, indices, updates, 0)`.
/// Implemented by iterating over unique index values and summing the corresponding updates.
fn scatter_add_1d(
    a: &Array,
    indices: &Array,
    updates: &Array,
) -> Result<Array, Exception> {
    // For each unique index position, sum the updates at that position and add to a.
    // We do this by creating one-hot encodings and using matrix operations.
    let n = a.shape()[0];
    let m = indices.shape()[0];

    // Create a one-hot matrix: (n, m) where one_hot[j, i] = 1 if indices[i] == j
    let arange_n = Array::arange::<_, i32>(None, n, None)?.reshape(&[n, 1])?;
    let indices_row = indices.reshape(&[1, m])?;
    let mask = arange_n.eq(&indices_row)?;
    let mask_f32 = mask.as_type::<f32>()?;

    // Scatter sum: for each position j, sum updates where indices == j
    // result[j] = sum(updates[i] for i where indices[i] == j) = mask_f32[j, :] @ updates
    let updates_col = updates.reshape(&[m, 1])?;
    let sums = mask_f32.matmul(&updates_col)?; // (n, 1)
    let sums = sums.reshape(&[n])?;

    Ok(a + &sums)
}

/// Generate a Hanning window of the given size.
///
/// If `periodic` is true, the denominator is `size` (for spectral analysis).
/// Otherwise it is `size - 1` (symmetric window).
pub fn hanning(size: i32, periodic: bool) -> Result<Array, Exception> {
    let denom = if periodic { size } else { size - 1 } as f64;
    let values: Vec<f32> = (0..size)
        .map(|n| (0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / denom).cos())) as f32)
        .collect();
    Ok(Array::from_slice(&values, &[size]))
}

/// Compute the Short-Time Fourier Transform.
///
/// - `x`: 1-D input signal
/// - `n_fft`: FFT size (default 800)
/// - `hop_length`: hop between frames (default n_fft / 4)
/// - `win_length`: window length (default n_fft)
/// - `center`: whether to center-pad the signal with reflection
pub fn stft(
    x: &Array,
    n_fft: Option<i32>,
    hop_length: Option<i32>,
    win_length: Option<i32>,
    center: Option<bool>,
) -> Result<Array, Exception> {
    let n_fft = n_fft.unwrap_or(800);
    let hop_length = hop_length.unwrap_or(n_fft / 4);
    let win_length = win_length.unwrap_or(n_fft);
    let center = center.unwrap_or(true);

    let w = hanning(win_length, false)?;

    // Pad window to n_fft if needed
    let w = if win_length < n_fft {
        let pad_len = n_fft - win_length;
        let pad = Array::zeros::<f32>(&[pad_len])?;
        mlx_rs::ops::concatenate_axis(&[&w, &pad], 0)?
    } else {
        w
    };

    // Center padding with reflect
    let x = if center {
        let half = n_fft / 2;
        let x_len = x.shape()[0];

        // prefix = x[1 : half+1] reversed
        let prefix_indices: Vec<i32> = (1..=half).rev().collect();
        let prefix_idx = Array::from_slice(&prefix_indices, &[half]);
        let prefix = x.index(&prefix_idx);

        // suffix = x[-(half+1) : -1] reversed
        let suffix_indices: Vec<i32> = ((x_len - half - 1)..=(x_len - 2)).rev().collect();
        let suffix_idx = Array::from_slice(&suffix_indices, &[half]);
        let suffix = x.index(&suffix_idx);

        mlx_rs::ops::concatenate_axis(&[&prefix, x, &suffix], 0)?
    } else {
        x.clone()
    };

    let x_len = x.shape()[0];
    let num_frames = 1 + (x_len - n_fft) / hop_length;

    // Create strided view of the frames
    let shape = [num_frames, n_fft];
    let strides = [hop_length as i64, 1i64];
    let frames = x.as_strided(&shape, &strides, 0)?;

    // Window and FFT (rfft along last axis by default)
    let windowed = frames * &w;
    mlx_rs::fft::rfft(&windowed, None, None)
}

/// Compute the inverse Short-Time Fourier Transform.
///
/// - `x`: complex STFT tensor of shape (freq_bins, num_frames)
/// - `hop_length`: hop between frames
/// - `win_length`: window length
/// - `center`: whether the original signal was center-padded
/// - `length`: optional output length to truncate/pad to
/// - `normalized`: if true, use squared window for normalization
pub fn istft(
    x: &Array,
    hop_length: Option<i32>,
    win_length: Option<i32>,
    center: Option<bool>,
    length: Option<i32>,
    normalized: Option<bool>,
) -> Result<Array, Exception> {
    let freq_bins = x.shape()[1];
    let win_length = win_length.unwrap_or((freq_bins - 1) * 2);
    let hop_length = hop_length.unwrap_or(win_length / 4);
    let center = center.unwrap_or(true);
    let normalized = normalized.unwrap_or(false);

    // Window: hanning(win_length + 1)[:-1]
    let w_full = hanning(win_length + 1, false)?;
    let w = w_full.index(0..win_length);

    let num_frames = x.shape()[1];
    let t = (num_frames - 1) * hop_length + win_length;

    // irfft along axis 0, then transpose to (num_frames, win_length)
    let frames_time = mlx_rs::fft::irfft(x, None, 0)?;
    let frames_time = frames_time.transpose_axes(&[1, 0])?;

    // Overlap-add using scatter_add_single
    let total_indices = num_frames * win_length;
    let mut all_indices = Vec::with_capacity(total_indices as usize);
    let mut recon_updates = Vec::with_capacity(num_frames as usize);
    let mut window_updates = Vec::with_capacity(num_frames as usize);

    let window_norm = if normalized {
        &w * &w
    } else {
        w.clone()
    };

    for f in 0..num_frames {
        let offset = f * hop_length;
        all_indices.extend(offset..offset + win_length);

        let frame = frames_time.index(f);
        recon_updates.push(&frame * &w);
        window_updates.push(window_norm.clone());
    }

    let indices = Array::from_slice(&all_indices, &[total_indices]);

    let recon_refs: Vec<&Array> = recon_updates.iter().collect();
    let recon_flat = mlx_rs::ops::concatenate_axis(&recon_refs, 0)?;

    let win_refs: Vec<&Array> = window_updates.iter().collect();
    let win_flat = mlx_rs::ops::concatenate_axis(&win_refs, 0)?;

    let reconstructed = Array::zeros::<f32>(&[t])?;
    let window_sum = Array::zeros::<f32>(&[t])?;

    // Overlap-add: accumulate into reconstructed and window_sum using put_along_axis
    // We use a manual approach since scatter_add_single is not available in mlx-rs 0.25.x
    let reconstructed = scatter_add_1d(&reconstructed, &indices, &recon_flat)?;
    let window_sum = scatter_add_1d(&window_sum, &indices, &win_flat)?;

    // Avoid division by zero
    let eps = Array::from_slice(&[1e-10f32], &[1]);
    let mask = window_sum.gt(&eps)?;
    let ones = Array::from_slice(&[1.0f32], &[1]);
    let safe_window = mlx_rs::ops::r#where(&mask, &window_sum, &ones)?;
    let reconstructed = reconstructed / safe_window;

    let reconstructed = if center && length.is_none() {
        let half = win_length / 2;
        let end = t - half;
        reconstructed.index(half..end)
    } else {
        reconstructed
    };

    let reconstructed = if let Some(len) = length {
        reconstructed.index(0..len)
    } else {
        reconstructed
    };

    Ok(reconstructed)
}

// ---------------------------------------------------------------------------
// MlxStft: convenience wrapper used by the vocoder generator
// ---------------------------------------------------------------------------

/// STFT/iSTFT wrapper that stores filter parameters and provides `transform`/`inverse`
/// methods for the vocoder pipeline.
#[derive(Debug)]
pub struct MlxStft {
    pub n_fft: i32,
    pub hop_size: i32,
    pub win_size: i32,
}

impl MlxStft {
    pub fn new(n_fft: i32, hop_size: i32, win_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            n_fft,
            hop_size,
            win_size,
        })
    }

    /// Compute STFT magnitude and phase for a batch of signals.
    ///
    /// Input: `x` of shape `(B, T)`.
    /// Returns `(magnitude, phase)` each of shape `(B, n_fft/2+1, num_frames)`.
    pub fn transform(&self, x: &Array) -> Result<(Array, Array), Exception> {
        let batch_size = x.shape()[0];
        let mut mags = Vec::with_capacity(batch_size as usize);
        let mut phases = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size {
            let signal = x.index(b); // (T,)
            let spec = stft(
                &signal,
                Some(self.n_fft),
                Some(self.hop_size),
                Some(self.win_size),
                Some(true),
            )?;

            // spec: (num_frames, n_fft/2+1) complex
            let mag = spec.abs()?;
            let phase = mlx_angle(&spec)?;

            // Transpose to (n_fft/2+1, num_frames)
            let mag = mag.transpose_axes(&[1, 0])?;
            let phase = phase.transpose_axes(&[1, 0])?;

            mags.push(mag);
            phases.push(phase);
        }

        let mag_refs: Vec<&Array> = mags.iter().collect();
        let phase_refs: Vec<&Array> = phases.iter().collect();

        let mag_batch = mlx_rs::ops::stack_axis(&mag_refs, 0)?;
        let phase_batch = mlx_rs::ops::stack_axis(&phase_refs, 0)?;

        Ok((mag_batch, phase_batch))
    }

    /// Inverse STFT from magnitude and phase.
    ///
    /// Input: `spec` of shape `(B, n_fft/2+1, num_frames)` and `phase` same shape.
    /// Returns audio of shape `(B, T)`.
    pub fn inverse(&self, spec: &Array, phase: &Array) -> Result<Array, Exception> {
        // Reconstruct complex STFT: spec * exp(j * phase) = spec * (cos(phase) + j*sin(phase))
        let cos_phase = mlx_rs::ops::cos(phase)?;
        let sin_phase = mlx_rs::ops::sin(phase)?;
        let real = spec * &cos_phase;
        let imag = spec * &sin_phase;

        let batch_size = spec.shape()[0];
        let mut outputs = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size {
            let r = real.index(b); // (freq, frames)
            let im = imag.index(b); // (freq, frames)

            // Build complex array: transpose to (frames, freq)
            let r_t = r.transpose_axes(&[1, 0])?;
            let im_t = im.transpose_axes(&[1, 0])?;

            // For irfft we need a complex input. Since mlx-rs may not directly support
            // creating complex arrays from real/imag, we use the real part only with
            // the phase encoding already applied. We reconstruct via:
            // output = sum of shifted windowed cosines (overlap-add approach).
            //
            // Actually, irfft expects complex input. We need to combine real and imag
            // into a complex array. Use view trick: interleave real and imag as float pairs.
            // complex64 = [f32 real, f32 imag] per element.
            let r_flat = r_t.reshape(&[-1, 1])?;
            let im_flat = im_t.reshape(&[-1, 1])?;
            let interleaved = mlx_rs::ops::concatenate_axis(&[&r_flat, &im_flat], -1)?;
            let interleaved = interleaved.reshape(&[r_t.shape()[0], r_t.shape()[1], 2])?;

            // View as complex64: (frames, freq, 2) float32 -> (frames, freq) complex64
            let complex_spec = interleaved
                .reshape(&[r_t.shape()[0], r_t.shape()[1] * 2])?
                .view::<mlx_rs::complex64>()?;

            // istft operates on (freq, frames) so transpose
            let complex_spec_t = complex_spec.transpose_axes(&[1, 0])?;

            let audio = istft(
                &complex_spec_t,
                Some(self.hop_size),
                Some(self.win_size),
                Some(true),
                None,
                None,
            )?;
            outputs.push(audio);
        }

        let out_refs: Vec<&Array> = outputs.iter().collect();
        mlx_rs::ops::stack_axis(&out_refs, 0)
    }
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Interpolate a 3-D tensor (N, C, W) along the spatial dimension by a scale factor.
///
/// This is the simple 3-argument form used by the vocoder source module.
/// `scale_factor` is a single float that scales the width dimension.
/// Supports "nearest" and "linear" modes.
pub fn interpolate(
    input: &Array,
    scale_factor: f32,
    mode: &str,
) -> Result<Array, Exception> {
    let in_w = input.shape()[2] as f32;
    let target_size = (in_w * scale_factor).ceil().max(1.0) as i32;
    interpolate1d(input, target_size, mode, None)
}

/// Interpolate a 3-D tensor (N, C, W) with full options (size or scale_factor).
pub fn interpolate_with_options(
    input: &Array,
    size: Option<&[i32]>,
    scale_factor: Option<&[f32]>,
    mode: &str,
    align_corners: Option<bool>,
) -> Result<Array, Exception> {
    let target_size = if let Some(sz) = size {
        sz[0]
    } else if let Some(sf) = scale_factor {
        let in_w = input.shape()[2] as f32;
        (in_w * sf[0]).ceil().max(1.0) as i32
    } else {
        return Err(Exception::custom(
            "interpolate requires either size or scale_factor",
        ));
    };
    interpolate1d(input, target_size, mode, align_corners)
}

/// 1-D interpolation of a (N, C, W) tensor to a new width.
pub fn interpolate1d(
    input: &Array,
    size: i32,
    mode: &str,
    align_corners: Option<bool>,
) -> Result<Array, Exception> {
    let in_width = input.shape()[2];
    let align_corners = align_corners.unwrap_or(false);

    if mode == "nearest" {
        let scale = in_width as f32 / size as f32;
        let arange = Array::arange::<_, f32>(None, size, None)?;
        let scale_arr = Array::from_slice(&[scale], &[1]);
        let indices = (&arange * &scale_arr).floor()?;
        let indices = indices.as_type::<i32>()?;
        let zero = Array::from_slice(&[0i32], &[1]);
        let max_idx = Array::from_slice(&[in_width - 1], &[1]);
        let indices = mlx_rs::ops::clip(&indices, (&zero, &max_idx))?;
        Ok(input.index((Ellipsis, &indices)))
    } else {
        // Linear interpolation
        let x = if align_corners && size > 1 {
            let scale = (in_width - 1) as f32 / (size - 1) as f32;
            let arange = Array::arange::<_, f32>(None, size, None)?;
            let scale_arr = Array::from_slice(&[scale], &[1]);
            &arange * &scale_arr
        } else {
            let scale = in_width as f32 / size as f32;
            let arange = Array::arange::<_, f32>(None, size, None)?;
            let scale_arr = Array::from_slice(&[scale], &[1]);
            let offset = Array::from_slice(&[0.5f32 * scale - 0.5f32], &[1]);
            &arange * &scale_arr + &offset
        };

        let x_low = x.floor()?.as_type::<i32>()?;
        let x_low_plus_one = &x_low + Array::from_slice(&[1i32], &[1]);
        let max_idx = Array::from_slice(&[in_width - 1], &[1]);
        let x_high = mlx_rs::ops::minimum(&x_low_plus_one, &max_idx)?;

        let x_low_f = x_low.as_type::<f32>()?;
        let x_frac = &x - &x_low_f;

        let y_low = input.index((Ellipsis, &x_low));
        let y_high = input.index((Ellipsis, &x_high));

        let one = Array::from_slice(&[1.0f32], &[1]);
        let one_minus_frac = (&one - &x_frac).reshape(&[1, 1, size])?;
        let x_frac = x_frac.reshape(&[1, 1, size])?;

        let output = y_low * one_minus_frac + y_high * x_frac;
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Check if an array has shape (out_channels, kH, kW) where
/// out_channels >= kH, out_channels >= kW, and kH == kW.
pub fn check_array_shape(arr: &Array) -> bool {
    let shape = arr.shape();
    if shape.len() != 3 {
        return false;
    }
    let out_channels = shape[0];
    let kh = shape[1];
    let kw = shape[2];
    out_channels >= kh && out_channels >= kw && kh == kw
}

/// Compute the angle (argument) of a complex array, or zero for real arrays.
pub fn mlx_angle(z: &Array) -> Result<Array, Exception> {
    use mlx_rs::Dtype;
    if z.dtype() == Dtype::Complex64 {
        let zimag = z.imag()?;
        let zreal = z.real()?;
        mlx_rs::ops::atan2(&zimag, &zreal)
    } else {
        let zimag = mlx_rs::ops::zeros_like(z)?;
        mlx_rs::ops::atan2(&zimag, z)
    }
}

/// Unwrap a phase array by changing absolute jumps greater than `discont` to their
/// `2*pi` complement along `axis`.
///
/// Equivalent to `numpy.unwrap`.
pub fn mlx_unwrap(
    p: &Array,
    discont: Option<f64>,
    axis: Option<i32>,
    period: Option<f64>,
) -> Result<Array, Exception> {
    let period = period.unwrap_or(2.0 * std::f64::consts::PI);
    let discont = discont.unwrap_or(period / 2.0).max(period / 2.0);
    let axis = axis.unwrap_or(-1);
    let ndim = p.ndim() as i32;
    let resolved_axis = if axis < 0 { ndim + axis } else { axis };

    let interval_high = period / 2.0;
    let interval_low = -interval_high;

    let dim_size = p.shape()[resolved_axis as usize];

    let p_after = slice_along_axis(p, resolved_axis, 1, dim_size)?;
    let p_before = slice_along_axis(p, resolved_axis, 0, dim_size - 1)?;
    let dd = &p_after - &p_before;

    let period_arr = Array::from_slice(&[period as f32], &[1]);
    let interval_low_arr = Array::from_slice(&[interval_low as f32], &[1]);
    let interval_high_arr = Array::from_slice(&[interval_high as f32], &[1]);
    let discont_arr = Array::from_slice(&[discont as f32], &[1]);

    let shifted = &dd - &interval_low_arr;
    let floored = (&shifted / &period_arr).floor()?;
    let ddmod = &dd - &(&period_arr * &floored);

    // Handle boundary: where |dd - interval_high| < eps and dd > 0, use interval_high
    let diff_from_high = (&dd - &interval_high_arr).abs()?;
    let eps = Array::from_slice(&[1e-10f32], &[1]);
    let zero = Array::from_slice(&[0.0f32], &[1]);
    let close_to_high = diff_from_high.lt(&eps)?;
    let dd_positive = dd.gt(&zero)?;
    let condition = close_to_high.logical_and(&dd_positive)?;
    let ddmod = mlx_rs::ops::r#where(&condition, &interval_high_arr, &ddmod)?;

    let ph_correct = &ddmod - &dd;

    // Zero out corrections where jump is small
    let abs_dd = dd.abs()?;
    let small_jump = abs_dd.lt(&discont_arr)?;
    let ph_correct = mlx_rs::ops::r#where(&small_jump, &zero, &ph_correct)?;

    // Pad with zeros along axis, then cumulative sum
    let mut pad_shape: Vec<i32> = p.shape().to_vec();
    pad_shape[resolved_axis as usize] = 1;
    let zero_padding = Array::zeros::<f32>(&pad_shape)?;

    let padded = mlx_rs::ops::concatenate_axis(&[&zero_padding, &ph_correct], resolved_axis)?;
    let cumulative = padded.cumsum(resolved_axis, None, None)?;

    Ok(p + cumulative)
}

/// Slice an array along a given axis from `start` to `stop`.
fn slice_along_axis(
    arr: &Array,
    axis: i32,
    start: i32,
    stop: i32,
) -> Result<Array, Exception> {
    let ndim = arr.ndim();
    let axis_usize = if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    };

    use mlx_rs::ops::indexing::ArrayIndexOp;
    let mut ops: Vec<ArrayIndexOp> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        if i == axis_usize {
            ops.push((start..stop).index_op());
        } else {
            ops.push((..).index_op());
        }
    }
    arr.try_index(&ops[..])
}
