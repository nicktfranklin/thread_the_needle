def serialize_object(obj):
    if isinstance(obj, (list, tuple)):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):  # Check for custom objects
        data = (
            obj.__dict__.copy()
        )  # Create a copy to avoid modifying the original object
        for key, value in data.items():
            print(key, type(key))
            data[key] = serialize_object(value)  # Recursively serialize nested objects
        return data
    else:
        return obj
