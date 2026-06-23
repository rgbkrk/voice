use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::{Result, VoxtralError};

pub const VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM: usize = 3072;

const BF16_BYTES: usize = 2;
const VOICE_EMBEDDING_DATA_PATHS: &[&str] = &["voice_embed/data/0", "archive/data/0", "data/0"];

const LOCAL_FILE_HEADER_SIGNATURE: u32 = 0x0403_4b50;
const CENTRAL_DIRECTORY_HEADER_SIGNATURE: u32 = 0x0201_4b50;
const END_OF_CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x0605_4b50;
const ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
const ZIP64_END_OF_CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x0606_4b50;
const ZIP_STORED: u16 = 0;
const ZIP64_EXTRA_FIELD_ID: u16 = 0x0001;

/// Load an official Voxtral voice embedding `.pt` file.
///
/// Voxtral voice embedding files are PyTorch zip archives that contain one raw
/// BF16 tensor storage under `data/0`, usually rooted at `voice_embed/` or
/// `archive/`. The tensor is expected to have hidden dimension 3072.
pub fn load_voice_embedding(
    path: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    load_voice_embedding_with_hidden_dim(path, VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM, dtype, device)
}

pub fn load_voice_embedding_with_hidden_dim(
    path: impl AsRef<Path>,
    hidden_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let bytes = fs::read(path)?;
    load_voice_embedding_bytes_with_hidden_dim(&bytes, hidden_dim, dtype, device)
}

fn load_voice_embedding_bytes_with_hidden_dim(
    archive: &[u8],
    hidden_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let data = extract_voice_embedding_data(archive)?;
    voice_embedding_tensor_from_bf16_bytes(data, hidden_dim, dtype, device)
}

fn voice_embedding_tensor_from_bf16_bytes(
    data: &[u8],
    hidden_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if hidden_dim == 0 {
        return Err(VoxtralError::InvalidCheckpoint(
            "voice embedding hidden dim must be non-zero".into(),
        ));
    }
    if data.is_empty() {
        return Err(VoxtralError::InvalidCheckpoint(
            "voice embedding tensor data is empty".into(),
        ));
    }
    if data.len() % BF16_BYTES != 0 {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "voice embedding BF16 data has bad byte length {}: expected an even number of bytes",
            data.len()
        )));
    }

    let row_bytes = hidden_dim.checked_mul(BF16_BYTES).ok_or_else(|| {
        VoxtralError::InvalidCheckpoint(format!(
            "voice embedding hidden dim {hidden_dim} is too large"
        ))
    })?;
    if data.len() % row_bytes != 0 {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "voice embedding BF16 data has bad byte length {}: expected a multiple of {row_bytes} bytes for hidden dim {hidden_dim}",
            data.len()
        )));
    }

    let rows = data.len() / row_bytes;
    Tensor::from_raw_buffer(data, DType::BF16, &[rows, hidden_dim], &Device::Cpu)
        .and_then(|tensor| tensor.to_dtype(dtype))
        .and_then(|tensor| tensor.to_device(device))
        .map_err(|e| VoxtralError::Candle(e.to_string()))
}

fn extract_voice_embedding_data(archive: &[u8]) -> Result<&[u8]> {
    let entries = parse_zip_entries(archive)?;
    let entry = find_voice_embedding_data_entry(&entries)?;
    extract_stored_zip_entry(archive, entry)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ZipEntry {
    name: String,
    compression_method: u16,
    compressed_size: u64,
    uncompressed_size: u64,
    local_header_offset: u64,
}

fn find_voice_embedding_data_entry(entries: &[ZipEntry]) -> Result<&ZipEntry> {
    for candidate in VOICE_EMBEDDING_DATA_PATHS {
        if let Some(entry) = entries
            .iter()
            .find(|entry| path_matches_candidate(&entry.name, candidate))
        {
            return Ok(entry);
        }
    }

    let mut data_zero_entries = entries
        .iter()
        .filter(|entry| entry.name == "data/0" || entry.name.ends_with("/data/0"));
    match (data_zero_entries.next(), data_zero_entries.next()) {
        (Some(entry), None) => Ok(entry),
        (Some(_), Some(_)) => Err(VoxtralError::InvalidCheckpoint(
            "found multiple data/0 entries in voice embedding archive; expected one BF16 tensor data entry".into(),
        )),
        (None, _) => Err(VoxtralError::InvalidCheckpoint(format!(
            "missing BF16 voice embedding tensor data; looked for {}",
            VOICE_EMBEDDING_DATA_PATHS.join(", ")
        ))),
    }
}

fn path_matches_candidate(name: &str, candidate: &str) -> bool {
    if name == candidate {
        return true;
    }
    let Some(prefix) = name.strip_suffix(candidate) else {
        return false;
    };
    prefix.ends_with('/')
}

fn parse_zip_entries(archive: &[u8]) -> Result<Vec<ZipEntry>> {
    let eocd_offset = find_end_of_central_directory(archive)?;
    let eocd = checked_slice(archive, eocd_offset, 22, "ZIP end-of-central-directory")?;
    let disk_number = read_u16(eocd, 4, "ZIP end-of-central-directory disk number")?;
    let central_directory_disk = read_u16(
        eocd,
        6,
        "ZIP end-of-central-directory central directory disk",
    )?;
    if disk_number != 0 || central_directory_disk != 0 {
        return Err(VoxtralError::InvalidCheckpoint(
            "multi-disk zip archives are not supported for Voxtral voice embeddings".into(),
        ));
    }

    let entries_16 = read_u16(eocd, 10, "ZIP central directory entry count")?;
    let central_size_32 = read_u32(eocd, 12, "ZIP central directory size")?;
    let central_offset_32 = read_u32(eocd, 16, "ZIP central directory offset")?;
    let (entry_count, central_size, central_offset) =
        if entries_16 == u16::MAX || central_size_32 == u32::MAX || central_offset_32 == u32::MAX {
            read_zip64_central_directory_info(archive, eocd_offset)?
        } else {
            (
                u64::from(entries_16),
                u64::from(central_size_32),
                u64::from(central_offset_32),
            )
        };

    let central_offset = usize_from_u64(central_offset, "ZIP central directory offset")?;
    let central_size = usize_from_u64(central_size, "ZIP central directory size")?;
    let central = checked_slice(
        archive,
        central_offset,
        central_size,
        "ZIP central directory",
    )?;

    let mut entries = Vec::new();
    let mut offset = 0usize;
    for _ in 0..entry_count {
        let header = checked_slice(central, offset, 46, "ZIP central directory file header")?;
        let signature = read_u32(header, 0, "ZIP central directory file header signature")?;
        if signature != CENTRAL_DIRECTORY_HEADER_SIGNATURE {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "invalid ZIP central directory file header at offset {offset}"
            )));
        }

        let compression_method = read_u16(header, 10, "ZIP compression method")?;
        let mut compressed_size = u64::from(read_u32(header, 20, "ZIP compressed size")?);
        let mut uncompressed_size = u64::from(read_u32(header, 24, "ZIP uncompressed size")?);
        let name_len = read_u16(header, 28, "ZIP file name length")? as usize;
        let extra_len = read_u16(header, 30, "ZIP extra field length")? as usize;
        let comment_len = read_u16(header, 32, "ZIP file comment length")? as usize;
        let mut local_header_offset = u64::from(read_u32(header, 42, "ZIP local header offset")?);
        let variable_len = name_len
            .checked_add(extra_len)
            .and_then(|n| n.checked_add(comment_len))
            .ok_or_else(|| {
                VoxtralError::InvalidCheckpoint(
                    "ZIP central directory variable fields are too large".into(),
                )
            })?;
        let variable = checked_slice(
            central,
            offset + 46,
            variable_len,
            "ZIP central directory variable fields",
        )?;
        let name_bytes = checked_slice(variable, 0, name_len, "ZIP file name")?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|e| {
                VoxtralError::InvalidCheckpoint(format!("ZIP entry name is not UTF-8: {e}"))
            })?
            .to_string();
        let extra = checked_slice(variable, name_len, extra_len, "ZIP extra fields")?;
        apply_zip64_extra(
            extra,
            &mut uncompressed_size,
            &mut compressed_size,
            &mut local_header_offset,
        )?;

        entries.push(ZipEntry {
            name,
            compression_method,
            compressed_size,
            uncompressed_size,
            local_header_offset,
        });
        offset = offset.checked_add(46 + variable_len).ok_or_else(|| {
            VoxtralError::InvalidCheckpoint("ZIP central directory offset overflow".into())
        })?;
    }

    if offset != central.len() {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "ZIP central directory has {} trailing bytes",
            central.len() - offset
        )));
    }

    Ok(entries)
}

fn extract_stored_zip_entry<'a>(archive: &'a [u8], entry: &ZipEntry) -> Result<&'a [u8]> {
    if entry.compression_method != ZIP_STORED {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "unsupported ZIP compression method {} for {}; expected uncompressed PyTorch tensor storage",
            entry.compression_method, entry.name
        )));
    }
    if entry.compressed_size != entry.uncompressed_size {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "stored ZIP entry {} has compressed size {} but uncompressed size {}",
            entry.name, entry.compressed_size, entry.uncompressed_size
        )));
    }

    let local_header_offset = usize_from_u64(entry.local_header_offset, "ZIP local header offset")?;
    let local_header = checked_slice(archive, local_header_offset, 30, "ZIP local file header")?;
    let signature = read_u32(local_header, 0, "ZIP local file header signature")?;
    if signature != LOCAL_FILE_HEADER_SIGNATURE {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "invalid ZIP local file header for {}",
            entry.name
        )));
    }
    let name_len = read_u16(local_header, 26, "ZIP local file name length")? as usize;
    let extra_len = read_u16(local_header, 28, "ZIP local extra field length")? as usize;
    let data_offset = local_header_offset
        .checked_add(30)
        .and_then(|offset| offset.checked_add(name_len))
        .and_then(|offset| offset.checked_add(extra_len))
        .ok_or_else(|| {
            VoxtralError::InvalidCheckpoint(format!(
                "ZIP local data offset overflow for {}",
                entry.name
            ))
        })?;
    let data_len = usize_from_u64(entry.uncompressed_size, "ZIP entry size")?;
    checked_slice(archive, data_offset, data_len, &entry.name)
}

fn find_end_of_central_directory(archive: &[u8]) -> Result<usize> {
    if archive.len() < 22 {
        return Err(VoxtralError::InvalidCheckpoint(
            "not a PyTorch zip archive: file is too short".into(),
        ));
    }

    let earliest = archive.len().saturating_sub(22 + u16::MAX as usize);
    let latest = archive.len() - 22;
    for offset in (earliest..=latest).rev() {
        if read_u32(archive, offset, "ZIP end-of-central-directory signature")?
            == END_OF_CENTRAL_DIRECTORY_SIGNATURE
        {
            let comment_len = read_u16(
                archive,
                offset + 20,
                "ZIP end-of-central-directory comment length",
            )? as usize;
            if offset + 22 + comment_len == archive.len() {
                return Ok(offset);
            }
        }
    }

    Err(VoxtralError::InvalidCheckpoint(
        "not a PyTorch zip archive: missing ZIP end-of-central-directory record".into(),
    ))
}

fn read_zip64_central_directory_info(
    archive: &[u8],
    eocd_offset: usize,
) -> Result<(u64, u64, u64)> {
    if eocd_offset < 20 {
        return Err(VoxtralError::InvalidCheckpoint(
            "missing ZIP64 end-of-central-directory locator".into(),
        ));
    }
    let locator_offset = eocd_offset - 20;
    let locator = checked_slice(
        archive,
        locator_offset,
        20,
        "ZIP64 end-of-central-directory locator",
    )?;
    let locator_signature = read_u32(
        locator,
        0,
        "ZIP64 end-of-central-directory locator signature",
    )?;
    if locator_signature != ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE {
        return Err(VoxtralError::InvalidCheckpoint(
            "missing ZIP64 end-of-central-directory locator".into(),
        ));
    }
    let zip64_eocd_offset = usize_from_u64(
        read_u64(locator, 8, "ZIP64 end-of-central-directory offset")?,
        "ZIP64 end-of-central-directory offset",
    )?;
    let zip64_eocd = checked_slice(
        archive,
        zip64_eocd_offset,
        56,
        "ZIP64 end-of-central-directory record",
    )?;
    let zip64_signature = read_u32(
        zip64_eocd,
        0,
        "ZIP64 end-of-central-directory record signature",
    )?;
    if zip64_signature != ZIP64_END_OF_CENTRAL_DIRECTORY_SIGNATURE {
        return Err(VoxtralError::InvalidCheckpoint(
            "invalid ZIP64 end-of-central-directory record".into(),
        ));
    }

    Ok((
        read_u64(zip64_eocd, 32, "ZIP64 central directory entry count")?,
        read_u64(zip64_eocd, 40, "ZIP64 central directory size")?,
        read_u64(zip64_eocd, 48, "ZIP64 central directory offset")?,
    ))
}

fn apply_zip64_extra(
    extra: &[u8],
    uncompressed_size: &mut u64,
    compressed_size: &mut u64,
    local_header_offset: &mut u64,
) -> Result<()> {
    if *uncompressed_size != u64::from(u32::MAX)
        && *compressed_size != u64::from(u32::MAX)
        && *local_header_offset != u64::from(u32::MAX)
    {
        return Ok(());
    }

    let mut offset = 0usize;
    while offset + 4 <= extra.len() {
        let header_id = read_u16(extra, offset, "ZIP extra header id")?;
        let data_len = read_u16(extra, offset + 2, "ZIP extra field size")? as usize;
        let data = checked_slice(extra, offset + 4, data_len, "ZIP extra field data")?;
        if header_id == ZIP64_EXTRA_FIELD_ID {
            let mut data_offset = 0usize;
            if *uncompressed_size == u64::from(u32::MAX) {
                *uncompressed_size = read_u64(data, data_offset, "ZIP64 uncompressed size")?;
                data_offset += 8;
            }
            if *compressed_size == u64::from(u32::MAX) {
                *compressed_size = read_u64(data, data_offset, "ZIP64 compressed size")?;
                data_offset += 8;
            }
            if *local_header_offset == u64::from(u32::MAX) {
                *local_header_offset = read_u64(data, data_offset, "ZIP64 local header offset")?;
            }
            return Ok(());
        }
        offset = offset.checked_add(4 + data_len).ok_or_else(|| {
            VoxtralError::InvalidCheckpoint("ZIP extra field offset overflow".into())
        })?;
    }

    Err(VoxtralError::InvalidCheckpoint(
        "missing ZIP64 extra field for large ZIP entry".into(),
    ))
}

fn checked_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    len: usize,
    context: &str,
) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| VoxtralError::InvalidCheckpoint(format!("{context} offset overflow")))?;
    bytes.get(offset..end).ok_or_else(|| {
        VoxtralError::InvalidCheckpoint(format!(
            "{context} extends past end of archive: offset {offset}, length {len}, archive length {}",
            bytes.len()
        ))
    })
}

fn read_u16(bytes: &[u8], offset: usize, context: &str) -> Result<u16> {
    let slice = checked_slice(bytes, offset, 2, context)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize, context: &str) -> Result<u32> {
    let slice = checked_slice(bytes, offset, 4, context)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(bytes: &[u8], offset: usize, context: &str) -> Result<u64> {
    let slice = checked_slice(bytes, offset, 8, context)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        VoxtralError::InvalidCheckpoint(format!("{context} {value} does not fit in usize"))
    })
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use candle_core::{DType, Device};

    use super::*;

    #[test]
    fn loads_voice_embed_data_zero_from_zip() {
        let bytes = stored_zip(&[("voice_embed/data/0", &bf16_rows(1, 0)[..])]);
        let tensor = load_voice_embedding_bytes_with_hidden_dim(
            &bytes,
            VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.dims(), &[1, VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM]);
        let values = tensor.to_vec2::<f32>().unwrap();
        assert_eq!(values[0][0], 1.0);
        assert_eq!(values[0][1], 2.0);
        assert_eq!(values[0][2], 3.0);
        assert_eq!(values[0][3], 4.0);
    }

    #[test]
    fn preserves_row_count_from_byte_length() {
        let bytes = stored_zip(&[("archive/data/0", &bf16_rows(3, 4)[..])]);
        let tensor = load_voice_embedding_bytes_with_hidden_dim(
            &bytes,
            VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM,
            DType::BF16,
            &Device::Cpu,
        )
        .unwrap();

        assert_eq!(tensor.dtype(), DType::BF16);
        assert_eq!(tensor.dims(), &[3, VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM]);
    }

    #[test]
    fn public_loader_reads_pt_file_from_disk() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "voice-voxtral-embedding-{}-{stamp}.pt",
            std::process::id()
        ));
        fs::write(&path, stored_zip(&[("data/0", &bf16_rows(1, 8)[..])])).unwrap();

        let tensor = load_voice_embedding(&path, DType::F32, &Device::Cpu).unwrap();

        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.dims(), &[1, VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM]);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reports_missing_tensor_data() {
        let bytes = stored_zip(&[("archive/data.pkl", b"metadata")]);

        let err = load_voice_embedding_bytes_with_hidden_dim(
            &bytes,
            VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap_err();

        assert!(matches!(err, VoxtralError::InvalidCheckpoint(_)));
        assert!(err
            .to_string()
            .contains("missing BF16 voice embedding tensor data"));
    }

    #[test]
    fn reports_bad_bf16_byte_length() {
        let bytes = stored_zip(&[("voice_embed/data/0", &[0u8; 7][..])]);

        let err = load_voice_embedding_bytes_with_hidden_dim(
            &bytes,
            VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap_err();

        assert!(matches!(err, VoxtralError::InvalidCheckpoint(_)));
        assert!(err.to_string().contains("bad byte length"));
    }

    fn bf16_rows(rows: usize, start: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM * BF16_BYTES);
        for index in 0..rows * VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM {
            let value = ((start + index) % 4 + 1) as f32;
            let raw = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&raw.to_le_bytes());
        }
        bytes
    }

    fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut central_directory = Vec::new();

        for (name, data) in entries {
            let local_header_offset = bytes.len() as u32;
            write_u32(&mut bytes, LOCAL_FILE_HEADER_SIGNATURE);
            write_u16(&mut bytes, 20);
            write_u16(&mut bytes, 0);
            write_u16(&mut bytes, ZIP_STORED);
            write_u16(&mut bytes, 0);
            write_u16(&mut bytes, 0);
            write_u32(&mut bytes, 0);
            write_u32(&mut bytes, data.len() as u32);
            write_u32(&mut bytes, data.len() as u32);
            write_u16(&mut bytes, name.len() as u16);
            write_u16(&mut bytes, 0);
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(data);

            write_u32(&mut central_directory, CENTRAL_DIRECTORY_HEADER_SIGNATURE);
            write_u16(&mut central_directory, 20);
            write_u16(&mut central_directory, 20);
            write_u16(&mut central_directory, 0);
            write_u16(&mut central_directory, ZIP_STORED);
            write_u16(&mut central_directory, 0);
            write_u16(&mut central_directory, 0);
            write_u32(&mut central_directory, 0);
            write_u32(&mut central_directory, data.len() as u32);
            write_u32(&mut central_directory, data.len() as u32);
            write_u16(&mut central_directory, name.len() as u16);
            write_u16(&mut central_directory, 0);
            write_u16(&mut central_directory, 0);
            write_u16(&mut central_directory, 0);
            write_u16(&mut central_directory, 0);
            write_u32(&mut central_directory, 0);
            write_u32(&mut central_directory, local_header_offset);
            central_directory.extend_from_slice(name.as_bytes());
        }

        let central_directory_offset = bytes.len() as u32;
        let central_directory_size = central_directory.len() as u32;
        bytes.extend_from_slice(&central_directory);
        write_u32(&mut bytes, END_OF_CENTRAL_DIRECTORY_SIGNATURE);
        write_u16(&mut bytes, 0);
        write_u16(&mut bytes, 0);
        write_u16(&mut bytes, entries.len() as u16);
        write_u16(&mut bytes, entries.len() as u16);
        write_u32(&mut bytes, central_directory_size);
        write_u32(&mut bytes, central_directory_offset);
        write_u16(&mut bytes, 0);
        bytes
    }

    fn write_u16(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
