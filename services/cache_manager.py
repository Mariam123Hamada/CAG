import torch 
from transformers.cache_utils import DynamicCache

# The steps here are as follows:
# 1.Initilize the cache manger 
# 2. When the user query comes, check if the cache is empty or not
# 3. If the cache is empty, then we will call the knowledge loader to load
# 4. If the cache is not empty, then we will use the cache to answer the user query
# 5. We Will send the Query and the cache to the LLM to get the answer
# 6. We will return the answer to the user and also update the cache with the new information if needed
# 7.apply the dynamic Loading .





class CacheManager:

    def __init__(self):
        self.kv_cache = None

    def has_cache(self):
        return self.kv_cache is not None

    def set_cache(self, cache):
        self.kv_cache = cache

    def get_cache(self):
        return self.kv_cache

    def clear_cache(self):
        self.kv_cache = None


cache_manager = CacheManager()          