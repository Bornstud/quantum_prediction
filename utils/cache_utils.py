"""Caching utilities for improved performance"""
import streamlit as st
from functools import wraps
import hashlib
import pickle
from typing import Any, Callable


def streamlit_cache_data(func: Callable) -> Callable:
    """
    Decorator for caching data with Streamlit's cache_data
    Automatically handles cache invalidation
    """
    @st.cache_data(ttl=3600, show_spinner=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def streamlit_cache_resource(func: Callable) -> Callable:
    """
    Decorator for caching resources (models, connections) with Streamlit's cache_resource
    """
    @st.cache_resource(show_spinner=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def hash_array(arr) -> str:
    """Generate hash for numpy array for caching"""
    return hashlib.md5(pickle.dumps(arr)).hexdigest()


class SimpleCache:
    """Simple in-memory cache for frequently accessed data"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with LRU eviction"""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_order.clear()


session_cache = SimpleCache(max_size=50)
