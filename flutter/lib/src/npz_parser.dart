import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Parser for numpy .npz files (ZIP archive of .npy arrays).
///
/// This is a minimal implementation sufficient for loading Kokoro voice files.
/// Voice files contain float32 arrays keyed by voice name.
class NpzParser {
  NpzParser._();

  /// Load an .npz file and return a map of name -> Float32List.
  ///
  /// Each entry is a flattened float32 array. The original shape information
  /// is available via [loadWithShapes].
  static Future<Map<String, Float32List>> load(String path) async {
    final result = await loadWithShapes(path);
    return result.map((k, v) => MapEntry(k, v.data));
  }

  /// Load an .npz file with shape information preserved.
  static Future<Map<String, NpyArray>> loadWithShapes(String path) async {
    final bytes = await File(path).readAsBytes();
    return parseBytes(bytes);
  }

  /// Parse .npz bytes in memory.
  static Map<String, NpyArray> parseBytes(Uint8List bytes) {
    final result = <String, NpyArray>{};

    // .npz is a ZIP file. Parse local file headers.
    int offset = 0;
    while (offset < bytes.length - 4) {
      // Look for local file header signature: PK\x03\x04
      if (bytes[offset] != 0x50 ||
          bytes[offset + 1] != 0x4B ||
          bytes[offset + 2] != 0x03 ||
          bytes[offset + 3] != 0x04) {
        break;
      }

      final compressionMethod = _readUint16LE(bytes, offset + 8);
      final compressedSize = _readUint32LE(bytes, offset + 18);
      final uncompressedSize = _readUint32LE(bytes, offset + 22);
      final nameLength = _readUint16LE(bytes, offset + 26);
      final extraLength = _readUint16LE(bytes, offset + 28);

      final nameStart = offset + 30;
      final name = utf8.decode(bytes.sublist(nameStart, nameStart + nameLength));
      final dataStart = nameStart + nameLength + extraLength;

      // Only handle stored (uncompressed) entries - numpy saves as stored by default
      if (compressionMethod == 0) {
        final data = bytes.sublist(dataStart, dataStart + uncompressedSize);
        if (name.endsWith('.npy')) {
          final arrayName = name.substring(0, name.length - 4);
          result[arrayName] = _parseNpy(data);
        }
      }

      offset = dataStart + compressedSize;
    }

    return result;
  }

  /// Parse a .npy file (numpy binary format).
  static NpyArray _parseNpy(Uint8List data) {
    // .npy format:
    // - Magic: \x93NUMPY
    // - Major version (1 byte), minor version (1 byte)
    // - Header length (2 bytes for v1, 4 bytes for v2)
    // - Header string (Python dict with 'descr', 'fortran_order', 'shape')
    // - Raw data

    if (data[0] != 0x93 || data[1] != 0x4E) {
      throw FormatException('Invalid .npy magic bytes');
    }

    final majorVersion = data[6];
    final int headerLen;
    final int dataOffset;

    if (majorVersion == 1) {
      headerLen = _readUint16LE(data, 8);
      dataOffset = 10 + headerLen;
    } else {
      headerLen = _readUint32LE(data, 8);
      dataOffset = 12 + headerLen;
    }

    final headerStr = utf8.decode(data.sublist(10, 10 + headerLen)).trim();

    // Parse dtype from header
    final descrMatch = RegExp(r"'descr':\s*'([^']+)'").firstMatch(headerStr);
    if (descrMatch == null) {
      throw FormatException('Cannot parse dtype from .npy header');
    }
    final descr = descrMatch.group(1)!;

    // Parse shape
    final shapeMatch = RegExp(r"'shape':\s*\(([^)]*)\)").firstMatch(headerStr);
    List<int> shape = [];
    if (shapeMatch != null) {
      final shapeStr = shapeMatch.group(1)!.trim();
      if (shapeStr.isNotEmpty) {
        shape = shapeStr
            .split(',')
            .map((s) => int.parse(s.trim()))
            .where((n) => n > 0)
            .toList();
      }
    }

    // Extract raw data bytes
    final rawData = data.sublist(dataOffset);

    // Convert based on dtype
    final endian = descr.startsWith('<') ? 'little' : (descr.startsWith('>') ? 'big' : 'native');
    final typeChar = descr.length >= 2 ? descr[descr.length - 2] : descr[descr.length - 1];
    final typeSize = descr.length >= 2 ? int.parse(descr[descr.length - 1]) : 1;

    if (typeChar == 'f' && typeSize == 4) {
      // float32
      Float32List floats;
      if (endian == 'little' || endian == 'native') {
        floats = rawData.buffer.asFloat32List(
          rawData.offsetInBytes,
          rawData.lengthInBytes ~/ 4,
        );
      } else {
        // Big endian - need to swap
        final swapped = Uint8List(rawData.length);
        for (int i = 0; i < rawData.length; i += 4) {
          swapped[i] = rawData[i + 3];
          swapped[i + 1] = rawData[i + 2];
          swapped[i + 2] = rawData[i + 1];
          swapped[i + 3] = rawData[i];
        }
        floats = swapped.buffer.asFloat32List();
      }
      return NpyArray(data: floats, shape: shape, dtype: 'float32');
    } else if (typeChar == 'f' && typeSize == 8) {
      // float64 -> convert to float32
      final doubles = rawData.buffer.asFloat64List(
        rawData.offsetInBytes,
        rawData.lengthInBytes ~/ 8,
      );
      final floats = Float32List(doubles.length);
      for (int i = 0; i < doubles.length; i++) {
        floats[i] = doubles[i].toDouble();
      }
      return NpyArray(data: floats, shape: shape, dtype: 'float64');
    } else {
      throw FormatException('Unsupported numpy dtype: $descr');
    }
  }

  static int _readUint16LE(Uint8List bytes, int offset) {
    return bytes[offset] | (bytes[offset + 1] << 8);
  }

  static int _readUint32LE(Uint8List bytes, int offset) {
    return bytes[offset] |
        (bytes[offset + 1] << 8) |
        (bytes[offset + 2] << 16) |
        (bytes[offset + 3] << 24);
  }
}

/// A numpy array with its data and shape.
class NpyArray {
  final Float32List data;
  final List<int> shape;
  final String dtype;

  const NpyArray({
    required this.data,
    required this.shape,
    required this.dtype,
  });

  @override
  String toString() => 'NpyArray(shape=$shape, dtype=$dtype, len=${data.length})';
}
