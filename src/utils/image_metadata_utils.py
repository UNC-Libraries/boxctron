def retry_after_stripping_metadata(image, operation):
  try:
    return operation()
  except TypeError:
    # Pillow can propagate malformed TIFF metadata into lazy loads and JPEG saves.
    for metadata_key in ('xmp', 'icc_profile', 'exif'):
      image.info.pop(metadata_key, None)
    return operation()
