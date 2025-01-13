def format_size(size_in_bytes: int) -> str:
    """
    Format a size in bytes to a human readable string with appropriate units.
    
    Args:
        size_in_bytes: Size in bytes
        
    Returns:
        Formatted string with appropriate unit (bytes, KB, MB, GB, TB)
    """
    # Define the units and their respective sizes
    units = ['bytes', 'KB', 'MB', 'GB', 'TB']
    size = float(size_in_bytes)
    unit_index = 0
    
    # Keep dividing by 1024 until we get to an appropriate unit
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # Format with 2 decimal places if not in bytes
    if unit_index == 0:
        return f"{size:,.0f} {units[unit_index]}"
    else:
        return f"{size:,.2f} {units[unit_index]}"