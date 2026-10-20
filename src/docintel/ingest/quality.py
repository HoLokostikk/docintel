SCAN_CHAR_THRESHOLD = 50

LARGE_IMAGE_AREA_RATIO = 0.8


def is_scanned(char_count: int, has_large_image: bool) -> bool:

    return char_count < SCAN_CHAR_THRESHOLD and has_large_image


def char_density(total_chars: int, total_area: float) -> float:

    if total_area <= 0:
        return 0.0
    return total_chars / total_area
